#!/usr/bin/env python3
"""LoRA fine-tuning script for Nous.

Reads training data exported by Nous (JSONL or Alpaca format) and fine-tunes
a local model using unsloth for 4-bit QLoRA. The resulting adapter can be
merged and served via Ollama.

Usage:
    python scripts/finetune.py --data training_data.jsonl --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
    python scripts/finetune.py --data training_data.jsonl --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit --merge --export-gguf q4_k_m

Requirements:
    pip install unsloth transformers trl datasets peft
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_training_data(path: str) -> list[dict]:
    """Load training data from JSONL or JSON format."""
    data = []
    path = Path(path)

    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        print(f"Error: unsupported format {path.suffix}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} training pairs from {path}")
    return data


def format_for_training(pairs: list[dict]) -> list[dict]:
    """Convert Nous training pairs to ChatML messages format."""
    formatted = []
    for pair in pairs:
        messages = []

        # System message
        system = pair.get("system", "You are Nous, a helpful AI assistant.")
        messages.append({"role": "system", "content": system})

        # User input
        user_input = pair.get("input", pair.get("instruction", ""))
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})

        # Assistant output
        output = pair.get("output", "")
        if not output:
            continue
        messages.append({"role": "assistant", "content": output})

        formatted.append({"messages": messages})

    print(f"Formatted {len(formatted)} valid training examples")
    return formatted


def train(
    data_path: str,
    model_name: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_seq_length: int = 2048,
    merge: bool = False,
    export_gguf: str = "",
):
    """Run LoRA fine-tuning with unsloth."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: unsloth not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Load and format data
    pairs = load_training_data(data_path)
    formatted = format_for_training(pairs)
    if len(formatted) < 5:
        print(f"Error: need at least 5 training examples, got {len(formatted)}", file=sys.stderr)
        sys.exit(1)

    # Load model with 4-bit quantization
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Prepare dataset
    def format_messages(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(formatted)
    dataset = dataset.map(format_messages)

    # Split: 90% train, 10% eval
    if len(dataset) >= 20:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=output_dir,
        save_strategy="epoch",
        report_to="none",
    )

    # Train
    print(f"Training for {epochs} epochs with {len(train_dataset)} examples...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    stats = trainer.train()
    print(f"Training complete. Loss: {stats.training_loss:.4f}")

    # Save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    # Optionally merge and export
    if merge or export_gguf:
        print("Merging adapter with base model...")
        merged_dir = os.path.join(output_dir, "merged")

        if export_gguf:
            # Export as GGUF for Ollama
            print(f"Exporting GGUF ({export_gguf})...")
            model.save_pretrained_gguf(
                merged_dir, tokenizer, quantization_method=export_gguf
            )
            gguf_path = os.path.join(merged_dir, f"unsloth.{export_gguf.upper()}.gguf")
            print(f"GGUF exported: {gguf_path}")
            print(f"\nTo use with Ollama:")
            print(f"  ollama create nous-tuned -f Modelfile")
            print(f"  # Modelfile should contain: FROM {gguf_path}")
        else:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            print(f"Merged model saved to {merged_dir}")

    # Save training stats
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "model": model_name,
            "epochs": epochs,
            "training_examples": len(train_dataset),
            "eval_examples": len(eval_dataset) if eval_dataset else 0,
            "final_loss": stats.training_loss,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
        }, f, indent=2)
    print(f"Stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Nous LoRA fine-tuning")
    parser.add_argument("--data", required=True, help="Path to training data (JSONL or JSON)")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                        help="Base model name (default: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit)")
    parser.add_argument("--output", default="./nous-lora-output",
                        help="Output directory (default: ./nous-lora-output)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length (default: 2048)")
    parser.add_argument("--merge", action="store_true", help="Merge adapter with base model after training")
    parser.add_argument("--export-gguf", default="", help="Export as GGUF with quantization (e.g., q4_k_m, q8_0)")

    args = parser.parse_args()
    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        merge=args.merge,
        export_gguf=args.export_gguf,
    )


if __name__ == "__main__":
    main()
