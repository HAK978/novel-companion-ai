"""
Generate training data for fine-tuning using a teacher model (Qwen2.5-72B).
V2: Supports lightnovel-crawler format, even sampling across chapters.

Usage:
    python generate_training_data_v2.py --source-dir /path/to/chapters --sample 500
"""

import json
import re
import argparse
import glob
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    return re.sub(r"\s+", " ", text).strip()


def load_teacher_model(model_name: str):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model: {model_name} (fp16, multi-GPU)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded across GPUs")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 2000) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_entity_extraction(model, tokenizer, chapter_text: str, chapter_num: int) -> dict:
    prompt = f"""You are a precise entity extractor. Extract all characters mentioned in this chapter.

For each character, provide:
- name: their primary name
- aliases: list of other names/titles they go by (empty list if none)
- role: one short phrase describing what they do in this chapter
- relationships: list of objects with "character" (name) and "type" (ally/enemy/mentor/family/rival/unknown)

Return ONLY a valid JSON array. No explanation, no markdown, just JSON.

Chapter {chapter_num} text:
{chapter_text[:6000]}

JSON array:"""

    output = generate(model, tokenizer, prompt)

    training_input = f"""Extract all characters from this chapter as a JSON array. For each character include: name, aliases, role, and relationships.

Chapter {chapter_num}:
{chapter_text[:4000]}

JSON array:"""

    return {"input": training_input, "output": output}


def generate_qa_pairs(model, tokenizer, chapter_text: str, chapter_num: int) -> list:
    prompt = f"""Read this chapter carefully and generate exactly 5 questions that a reader might ask about it.

For each question, provide the answer using ONLY information from this chapter text. Do not make up any information.

Format your response as a JSON array of objects, each with "question" and "answer" keys.

Chapter {chapter_num} text:
{chapter_text[:6000]}

JSON array of 5 Q&A pairs:"""

    output = generate(model, tokenizer, prompt)

    try:
        cleaned = output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        qa_pairs = json.loads(cleaned)
        if not isinstance(qa_pairs, list):
            return []
    except json.JSONDecodeError:
        return []

    examples = []
    for qa in qa_pairs:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        if not q or not a:
            continue

        training_input = f"""Based on the following passage, answer this question.

Passage:
{chapter_text[:3000]}

Question: {q}

Answer:"""

        examples.append({"input": training_input, "output": a.strip()})

    return examples


def generate_summary(model, tokenizer, chapter_text: str, chapter_num: int) -> dict:
    prompt = f"""Summarize this chapter in 150-200 words. Cover the key events, character actions, and plot developments. Be specific about what happens.

Chapter {chapter_num} text:
{chapter_text[:6000]}

Summary:"""

    output = generate(model, tokenizer, prompt)

    training_input = f"""Summarize the following chapter in 150-200 words. Cover key events, character actions, and plot developments.

Chapter {chapter_num}:
{chapter_text[:4000]}

Summary:"""

    return {"input": training_input, "output": output}


def load_lncrawl_chapters(source_dir: str) -> list:
    """Load chapters from lightnovel-crawler directory structure."""
    source_path = Path(source_dir)
    all_files = sorted(glob.glob(str(source_path / "**" / "*.json"), recursive=True))

    # Filter out meta.json
    chapter_files = [f for f in all_files if not f.endswith("meta.json")]

    chapters = []
    for f in chapter_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            content = data.get("content", "")
            if not content or len(content) < 200:
                continue
            chapters.append({
                "number": data.get("serial", len(chapters) + 1),
                "title": data.get("title", f"Chapter {len(chapters) + 1}"),
                "content": content,
            })
        except Exception:
            continue

    chapters.sort(key=lambda c: c["number"])
    return chapters


def sample_evenly(chapters: list, n: int) -> list:
    """Sample n chapters evenly across the full list."""
    total = len(chapters)
    if n >= total:
        return chapters
    step = total / n
    indices = [int(i * step) for i in range(n)]
    return [chapters[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Generate training data (v2)")
    parser.add_argument("--source-dir", required=True, help="Path to lncrawl chapter directory")
    parser.add_argument("--sample", type=int, default=500, help="Number of chapters to sample")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct", help="Teacher model")
    parser.add_argument("--output-dir", default="data_v2", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all chapters
    print(f"Loading chapters from {args.source_dir}...")
    all_chapters = load_lncrawl_chapters(args.source_dir)
    print(f"Found {len(all_chapters)} chapters")

    # Sample evenly
    sampled = sample_evenly(all_chapters, args.sample)
    print(f"Sampled {len(sampled)} chapters evenly across the novel")
    print(f"Range: ch.{sampled[0]['number']} to ch.{sampled[-1]['number']}")

    # Load teacher model
    model, tokenizer = load_teacher_model(args.model)

    entity_file = open(output_dir / "entity_extraction.jsonl", "w")
    qa_file = open(output_dir / "qa_pairs.jsonl", "w")
    summary_file = open(output_dir / "summaries.jsonl", "w")

    total_examples = 0
    failed = 0

    for i, chapter in enumerate(sampled):
        chapter_num = chapter["number"]
        title = chapter["title"]
        text = clean_html(chapter["content"])

        if len(text) < 100:
            print(f"  Skipping ch.{chapter_num} (too short)")
            continue

        print(f"[{i+1}/{len(sampled)}] Chapter {chapter_num}: {title}")

        # 1. Entity extraction
        try:
            entity_example = generate_entity_extraction(model, tokenizer, text, chapter_num)
            entity_file.write(json.dumps(entity_example) + "\n")
            entity_file.flush()
            total_examples += 1
            print(f"  entities done")
        except Exception as e:
            print(f"  entities failed: {e}")
            failed += 1

        # 2. Q&A pairs
        try:
            qa_examples = generate_qa_pairs(model, tokenizer, text, chapter_num)
            for ex in qa_examples:
                qa_file.write(json.dumps(ex) + "\n")
            qa_file.flush()
            total_examples += len(qa_examples)
            print(f"  {len(qa_examples)} Q&A pairs")
        except Exception as e:
            print(f"  Q&A failed: {e}")
            failed += 1

        # 3. Summary
        try:
            summary_example = generate_summary(model, tokenizer, text, chapter_num)
            summary_file.write(json.dumps(summary_example) + "\n")
            summary_file.flush()
            total_examples += 1
            print(f"  summary done")
        except Exception as e:
            print(f"  summary failed: {e}")
            failed += 1

        print()

    entity_file.close()
    qa_file.close()
    summary_file.close()

    print(f"\n{'='*50}")
    print(f"Training data generation complete!")
    print(f"Total examples: {total_examples}")
    print(f"Failed: {failed}")
    print(f"Files:")
    print(f"  {output_dir}/entity_extraction.jsonl")
    print(f"  {output_dir}/qa_pairs.jsonl")
    print(f"  {output_dir}/summaries.jsonl")


if __name__ == "__main__":
    main()
