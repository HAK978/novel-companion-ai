"""
Generate training data for fine-tuning using a teacher model (Qwen2.5-72B).
Loads the model in fp16 across multiple GPUs via HuggingFace transformers.

Usage:
    python generate_training_data.py [--chapters 50] [--model Qwen/Qwen2.5-72B-Instruct]
"""

import json
import re
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    return re.sub(r"\s+", " ", text).strip()


def load_teacher_model(model_name: str):
    """Load teacher model in fp16 across multiple GPUs."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model: {model_name} (fp16, multi-GPU)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map}")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 2000) -> str:
    """Generate text from the teacher model."""
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

    # Decode only the generated tokens (skip the input)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_entity_extraction(model, tokenizer, chapter_text: str, chapter_num: int) -> dict:
    """Generate an entity extraction training example."""
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
    """Generate Q&A training examples from a chapter."""
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
    """Generate a summarization training example."""
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


def main():
    parser = argparse.ArgumentParser(description="Generate training data for fine-tuning")
    parser.add_argument("--source", default="/home/hkute/Novel Info Giver/shadow_slave_test.json",
                        help="Path to chapter JSON file")
    parser.add_argument("--chapters", type=int, default=50, help="Number of chapters to process")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct", help="Teacher model name")
    parser.add_argument("--output-dir", default="data", help="Output directory for training data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chapters
    with open(args.source, "r", encoding="utf-8") as f:
        data = json.load(f)
    chapters = data["chapters"][:args.chapters]
    print(f"Loaded {len(chapters)} chapters")

    # Load teacher model
    model, tokenizer = load_teacher_model(args.model)

    entity_file = open(output_dir / "entity_extraction.jsonl", "w")
    qa_file = open(output_dir / "qa_pairs.jsonl", "w")
    summary_file = open(output_dir / "summaries.jsonl", "w")

    total_examples = 0

    for i, chapter in enumerate(chapters):
        chapter_num = chapter.get("number", i + 1)
        title = chapter.get("title", f"Chapter {chapter_num}")
        content = chapter.get("content", "")
        text = clean_html(content) if "<" in content else content

        if len(text) < 100:
            print(f"  Skipping ch.{chapter_num} (too short)")
            continue

        print(f"\n[{i+1}/{len(chapters)}] Chapter {chapter_num}: {title}")

        # 1. Entity extraction
        try:
            entity_example = generate_entity_extraction(model, tokenizer, text, chapter_num)
            entity_file.write(json.dumps(entity_example) + "\n")
            entity_file.flush()
            total_examples += 1
            print(f"  entity extraction done")
        except Exception as e:
            print(f"  entity extraction failed: {e}")

        # 2. Q&A pairs
        try:
            qa_examples = generate_qa_pairs(model, tokenizer, text, chapter_num)
            for ex in qa_examples:
                qa_file.write(json.dumps(ex) + "\n")
            qa_file.flush()
            total_examples += len(qa_examples)
            print(f"  {len(qa_examples)} Q&A pairs generated")
        except Exception as e:
            print(f"  Q&A generation failed: {e}")

        # 3. Summary
        try:
            summary_example = generate_summary(model, tokenizer, text, chapter_num)
            summary_file.write(json.dumps(summary_example) + "\n")
            summary_file.flush()
            total_examples += 1
            print(f"  summary done")
        except Exception as e:
            print(f"  summary failed: {e}")

    entity_file.close()
    qa_file.close()
    summary_file.close()

    print(f"\n{'='*50}")
    print(f"Training data generation complete!")
    print(f"Total examples: {total_examples}")
    print(f"Files:")
    print(f"  {output_dir}/entity_extraction.jsonl")
    print(f"  {output_dir}/qa_pairs.jsonl")
    print(f"  {output_dir}/summaries.jsonl")


if __name__ == "__main__":
    main()
