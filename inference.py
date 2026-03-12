"""
inference.py

ファインチューニング済みモデルを使ってLINEキャラクターと対話するCLIツール。
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
MAX_NEW_TOKENS = 256


def resolve_base_model(model_dir: Path, base_model: str | None) -> str:
    """
    ベースモデルを解決する。

    --base-model が指定されていない場合は adapter_config.json から自動取得。
    """
    if base_model:
        return base_model
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config, encoding="utf-8") as f:
            cfg = json.load(f)
        detected = cfg.get("base_model_name_or_path")
        if detected:
            print(f"ベースモデルを自動検出しました: {detected}")
            return detected
    raise ValueError(
        "ベースモデルを特定できませんでした。--base-model で明示的に指定してください。"
    )


def load_model(model_dir: str | Path, base_model: str | None = None):
    """LoRAアダプタを適用したモデルとトークナイザーを読み込む。"""
    model_dir = Path(model_dir)
    base_model = resolve_base_model(model_dir, base_model)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    return model, tokenizer


def chat(
    model,
    tokenizer,
    speaker: str,
    history: list[dict],
    user_input: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    ユーザー入力に対してモデルの返答を生成する。

    Parameters
    ----------
    model : PeftModel
        ファインチューニング済みモデル
    tokenizer : AutoTokenizer
        トークナイザー
    speaker : str
        模倣している話者名（システムプロンプトに使用）
    history : list[dict]
        これまでの会話履歴（{"role": ..., "content": ...}のリスト）
    user_input : str
        ユーザーの入力テキスト
    max_new_tokens : int
        生成する最大トークン数

    Returns
    -------
    str
        モデルが生成した返答テキスト
    """
    system_prompt = (
        f"あなたは{speaker}です。"
        f"{speaker}の話し方・口調・言葉づかいを忠実に再現して返答してください。"
        "不自然に丁寧にならず、実際のLINEの会話のように自然に話してください。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_interactive(
    model_dir: str | Path,
    speaker: str,
    base_model: str | None = None,
    max_history: int = 6,
) -> None:
    """
    対話型CLIループを起動する。

    Parameters
    ----------
    model_dir : str | Path
        LoRAアダプタのディレクトリ
    speaker : str
        模倣する話者名
    base_model : str | None
        ベースモデルのHuggingFace ID。省略時は adapter_config.json から自動取得。
    max_history : int
        保持する会話ターン数（メモリ節約のため上限を設定）
    """
    print(f"\nモデルを読み込み中: {model_dir}")
    model, tokenizer = load_model(model_dir, base_model)

    print(f"\n--- {speaker}との会話を開始します（終了: 'exit' または Ctrl+C）---\n")
    history: list[dict] = []

    while True:
        try:
            user_input = input("あなた: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n会話を終了します。")
            break

        if user_input.lower() in {"exit", "quit", "終了"}:
            print("会話を終了します。")
            break
        if not user_input:
            continue

        response = chat(model, tokenizer, speaker, history, user_input)
        print(f"{speaker}: {response}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # 履歴が長くなりすぎないように古いものを削除
        if len(history) > max_history * 2:
            history = history[-max_history * 2:]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    parser = ArgumentParser(description="ファインチューニング済みモデルで対話")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                        help="LoRAアダプタのディレクトリ")
    parser.add_argument("--base-model", default=None,
                        help="ベースモデルのHuggingFace ID（省略時はadapter_config.jsonから自動取得）")
    parser.add_argument("--speaker", required=True, help="模倣する話者名")
    parser.add_argument("--max-history", type=int, default=6,
                        help="保持する会話ターン数")
    args = parser.parse_args()

    run_interactive(
        model_dir=args.model_dir,
        speaker=args.speaker,
        base_model=args.base_model,
        max_history=args.max_history,
    )


if __name__ == "__main__":
    main()
