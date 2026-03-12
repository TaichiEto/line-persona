"""
train.py

QLoRAを使ってQwen2.5をLINEコーパスでファインチューニングする。

依存ライブラリ:
  transformers, peft, trl, bitsandbytes, datasets, accelerate
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# デフォルトのベースモデル（Qwen2.5-7B-Instruct）
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = Path(__file__).parent / "models"


def load_corpus(corpus_path: str | Path) -> Dataset:
    """JSONL形式のコーパスをHuggingFace Datasetとして読み込む。"""
    records = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def build_qlora_config() -> LoraConfig:
    """QLoRA設定を返す。"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                        # LoRAランク
        lora_alpha=32,               # スケーリング係数
        lora_dropout=0.05,
        bias="none",
        target_modules=[             # Qwen2.5のAttentionレイヤ
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def load_model_and_tokenizer(model_name: str):
    """量子化モデルとトークナイザーを読み込む。"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # gradient checkpointingと非互換のため無効化

    return model, tokenizer


def format_messages(example: dict, tokenizer) -> dict:
    """
    messagesリストをモデルのchat templateで文字列に変換する。

    SFTTrainerはデータセットの "text" カラムを使う。
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train(
    corpus_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    output_dir: str | Path = OUTPUT_DIR,
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_seq_length: int = 1024,
    learning_rate: float = 2e-4,
) -> None:
    """
    LoRAファインチューニングを実行する。

    Parameters
    ----------
    corpus_path : str | Path
        学習用コーパス（JSONL）のパス
    model_name : str
        HuggingFaceのモデルID（デフォルト: Qwen2.5-7B-Instruct）
    output_dir : str | Path
        モデルの保存先ディレクトリ
    num_epochs : int
        学習エポック数
    batch_size : int
        バッチサイズ（GPUメモリに応じて調整）
    grad_accum : int
        勾配蓄積ステップ数
    max_seq_length : int
        最大シーケンス長
    learning_rate : float
        学習率
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデルを読み込み中: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)

    lora_config = build_qlora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"コーパスを読み込み中: {corpus_path}")
    dataset = load_corpus(corpus_path)
    dataset = dataset.map(
        lambda ex: format_messages(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",            # wandb等を使う場合は変更
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )

    print("学習開始...")
    trainer.train()

    print(f"モデルを保存中: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print("完了!")


def main() -> None:
    parser = ArgumentParser(description="QLoRAでLINEコーパスをファインチューニング")
    parser.add_argument("--corpus", required=True, help="コーパスJSONLのパス")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ベースモデルのHuggingFace ID")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="出力ディレクトリ")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=2, help="バッチサイズ")
    parser.add_argument("--grad-accum", type=int, default=8, help="勾配蓄積ステップ数")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="最大シーケンス長")
    parser.add_argument("--lr", type=float, default=2e-4, help="学習率")
    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
