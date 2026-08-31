#!/usr/bin/env python3
"""Generate all three arms of the frozen OOD function-QA battery.

Each question receives one greedy, 64-token generation from vector context,
finetuned text context, and stock text context. Valid receipt prefixes can be
resumed without regenerating completed rows.
"""

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

EXPECTED_QUESTIONS_SHA256 = "fc887b7865d8a4c8794549322f149df84d799189b8d39c7c9706cb979fa79e2e"
EXPECTED_N = 1268
MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 64

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_questions(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == EXPECTED_N, f"question count {len(rows)} != {EXPECTED_N}"
    required = {"func_name", "code", "question", "gold", "intent", "answer_type"}
    for i, row in enumerate(rows):
        assert required <= row.keys(), f"question {i} missing {required - row.keys()}"
        assert row["intent"] == "F2", f"question {i} intent {row['intent']} != F2"
    return rows


def qa_prompt(row: dict) -> str:
    return f"\nQ: {row['question']}\nA:"


def extracted_predictions(raw: str) -> tuple[str, str]:
    # Preserve the historical first-line extraction and also provide a
    # termination-aware extraction that stops at a subsequent QA template.
    first_line = raw.split("\n", 1)[0].strip()[:200]
    cuts = [pos for marker in ("\n", "Q:", "A:") if (pos := raw.find(marker)) >= 0]
    corrected = raw[: min(cuts)].strip()[:200] if cuts else raw.strip()[:200]
    return first_line, corrected


def grader_record(row: dict) -> dict:
    rec = dict(row)
    rec["answer"] = row["gold"]
    return rec


def load_receipt_prefix(path: Path, arm: str, questions: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) <= len(questions), f"{path}: too many rows"
    for i, (receipt, q) in enumerate(zip(rows, questions)):
        assert receipt["index"] == i and receipt["arm"] == arm, f"{path}:{i}: order/arm mismatch"
        assert (receipt["func_name"], receipt["question"], receipt["gold"]) == (
            q["func_name"], q["question"], q["gold"]
        ), f"{path}:{i}: frozen question mismatch"
    return rows


def configure_generation(model, tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id


def receipt(row: dict, index: int, arm: str, raw: str, rt_grade) -> dict:
    first_line, corrected = extracted_predictions(raw)
    rec = grader_record(row)
    return {
        "index": index,
        "arm": arm,
        "func_name": row["func_name"],
        "intent": row["intent"],
        "answer_type": row["answer_type"],
        "question": row["question"],
        "gold": row["gold"],
        "raw_pred": raw,
        "first_line_pred": first_line,
        "corrected_pred": corrected,
        "ok_first_line": bool(rt_grade(first_line, rec)),
        "ok_corrected": bool(rt_grade(corrected, rec)),
    }


def write_generated_batch(
    path: Path,
    arm: str,
    questions: list[dict],
    start: int,
    outputs,
    tokenizer,
    rt_grade,
) -> tuple[int, int]:
    raw_ok = corrected_ok = 0
    with path.open("a", encoding="utf-8") as f:
        for offset, output in enumerate(outputs):
            index = start + offset
            raw = tokenizer.decode(output.tolist(), skip_special_tokens=True)
            row = receipt(questions[index], index, arm, raw, rt_grade)
            raw_ok += row["ok_first_line"]
            corrected_ok += row["ok_corrected"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return raw_ok, corrected_ok


def print_tally(path: Path, arm: str) -> None:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    raw_ok = sum(r["ok_first_line"] for r in rows)
    corrected_ok = sum(r["ok_corrected"] for r in rows)
    print(
        f"ARM_DONE {arm}: n={len(rows)} first_line={raw_ok}/{len(rows)} "
        f"({100 * raw_ok / len(rows):.3f}%) corrected={corrected_ok}/{len(rows)} "
        f"({100 * corrected_ok / len(rows):.3f}%)",
        flush=True,
    )


def build_vector_cache(comp, questions: list[dict], device: str, batch_size: int) -> dict[str, object]:
    import torch

    unique_codes = list(dict.fromkeys(row["code"] for row in questions))
    tokenizer = comp.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cache = {}
    boundary = comp.boundary_pair()
    print(f"vector compression: {len(unique_codes)} unique code strings", flush=True)
    with torch.no_grad():
        for start in range(0, len(unique_codes), batch_size):
            codes = unique_codes[start : start + batch_size]
            batch = tokenizer(codes, return_tensors="pt", padding=True).to(device)
            lengths = batch.attention_mask.sum(dim=1).tolist()
            vectors = comp(batch.input_ids, batch.attention_mask)
            for i, (code, length) in enumerate(zip(codes, lengths)):
                groups = math.ceil(length / comp.pooling_factor)
                value = vectors[i, :groups]
                if boundary is not None:
                    value = torch.cat([boundary[0][0], value, boundary[1][0]], dim=0)
                cache[code] = value
            done = min(start + batch_size, len(unique_codes))
            print(f"  compressed {done}/{len(unique_codes)}", flush=True)
    return cache


def run_vector_arm(
    questions: list[dict],
    output_dir: Path,
    full_state: Path,
    projector: Path,
    gen_batch: int,
    compress_batch: int,
    rt_grade,
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from compressor import Compressor, ft_load
    from compressor.genutil import batched_generate

    arm = "vec"
    path = output_dir / f"{arm}.jsonl"
    prefix = load_receipt_prefix(path, arm, questions)
    if len(prefix) == len(questions):
        print(f"{arm}: complete receipt already present; not regenerating", flush=True)
        print_tally(path, arm)
        return None

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is required for this evaluation"
    hidden = AutoConfig.from_pretrained(MODEL_NAME).hidden_size
    comp = Compressor(
        encoder_name=MODEL_NAME,
        decoder_hidden=hidden,
        pooling_factor=4,
        pooling="latent",
        boundary=True,
    ).to(device)
    projector_meta = ft_load.load_projector(comp.projector, projector, prefer="ema")
    encoder_meta = ft_load.load_encoder(comp, full_state)
    comp.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    decoder = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device).eval()
    decoder_meta = ft_load.load_decoder(decoder, full_state)
    configure_generation(decoder, tokenizer)
    embed = decoder.get_input_embeddings()
    print(
        "LOADS vec " + json.dumps({"projector": projector_meta, "encoder": encoder_meta, "decoder": decoder_meta}),
        flush=True,
    )

    cache = build_vector_cache(comp, questions, device, compress_batch)
    start_at = len(prefix)
    for start in range(start_at, len(questions), gen_batch):
        chunk = questions[start : start + gen_batch]
        contexts = []
        for row in chunk:
            prompt_ids = tokenizer(qa_prompt(row), add_special_tokens=False).input_ids
            prompt_embeds = embed(torch.tensor(prompt_ids, device=device)).to(cache[row["code"]].dtype)
            contexts.append(torch.cat([cache[row["code"]], prompt_embeds], dim=0))
        outputs = batched_generate(decoder, contexts, MAX_NEW_TOKENS, batch_size=gen_batch)
        write_generated_batch(path, arm, questions, start, outputs, tokenizer, rt_grade)
        print(f"  [{arm}] {min(start + len(chunk), len(questions))}/{len(questions)}", flush=True)
    print_tally(path, arm)

    del cache, comp
    gc.collect()
    torch.cuda.empty_cache()
    return decoder, tokenizer, ft_load


def run_text_arm(
    arm: str,
    questions: list[dict],
    output_dir: Path,
    decoder,
    tokenizer,
    gen_batch: int,
    rt_grade,
) -> None:
    import torch
    from compressor.genutil import batched_generate

    path = output_dir / f"{arm}.jsonl"
    prefix = load_receipt_prefix(path, arm, questions)
    if len(prefix) == len(questions):
        print(f"{arm}: complete receipt already present; not regenerating", flush=True)
        print_tally(path, arm)
        return

    device = "cuda"
    embed = decoder.get_input_embeddings()
    code_ids = {
        code: tokenizer(code, add_special_tokens=False).input_ids
        for code in dict.fromkeys(row["code"] for row in questions)
    }
    start_at = len(prefix)
    for start in range(start_at, len(questions), gen_batch):
        chunk = questions[start : start + gen_batch]
        contexts = []
        for row in chunk:
            ids = code_ids[row["code"]] + tokenizer(qa_prompt(row), add_special_tokens=False).input_ids
            contexts.append(embed(torch.tensor(ids, device=device)))
        outputs = batched_generate(decoder, contexts, MAX_NEW_TOKENS, batch_size=gen_batch)
        write_generated_batch(path, arm, questions, start, outputs, tokenizer, rt_grade)
        print(f"  [{arm}] {min(start + len(chunk), len(questions))}/{len(questions)}", flush=True)
    print_tally(path, arm)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=root)
    parser.add_argument(
        "--questions",
        type=Path,
        default=root / "receipts" / "ood_qa" / "questions_frozen.jsonl.gz",
    )
    parser.add_argument(
        "--full-state",
        type=Path,
        default=root / "weights" / "model.safetensors",
    )
    parser.add_argument(
        "--projector",
        type=Path,
        default=root / "weights" / "projector_ema.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "receipts" / "ood_qa",
    )
    parser.add_argument("--gen-batch", type=int, default=64)
    parser.add_argument("--compress-batch", type=int, default=16)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root.resolve()))

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions)
    question_sha = sha256_file(args.questions)
    assert question_sha == EXPECTED_QUESTIONS_SHA256, (
        f"frozen questions changed: {question_sha}"
    )
    print(
        f"frozen questions: n={len(questions)} sha256={question_sha} "
        f"input={args.questions}",
        flush=True,
    )

    from dataset.qa_filters import rt_grade

    result = run_vector_arm(
        questions,
        args.output_dir,
        args.full_state,
        args.projector,
        args.gen_batch,
        args.compress_batch,
        rt_grade,
    )
    if result is None:
        # A resumed complete vec arm still needs the finetuned decoder for ft_text.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from compressor import ft_load

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        decoder = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda").eval()
        ft_load.load_decoder(decoder, args.full_state)
        configure_generation(decoder, tokenizer)
    else:
        decoder, tokenizer, ft_load = result

    run_text_arm("ft_text", questions, args.output_dir, decoder, tokenizer, args.gen_batch, rt_grade)

    # Release the finetuned model and cached state before loading the stock model.
    del decoder
    ft_load._CACHE.clear()
    gc.collect()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    decoder = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda").eval()
    configure_generation(decoder, tokenizer)
    print("LOADS stock_text: stock Qwen/Qwen3-1.7B", flush=True)
    run_text_arm("stock_text", questions, args.output_dir, decoder, tokenizer, args.gen_batch, rt_grade)

    print(f"ALL_ARMS_DONE elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
