# line-qlora — LINEトーク履歴でAIを学習させる

LINEのトーク履歴をコーパスにして、特定の人物の口調を模倣するLLMをファインチューニングするプロジェクトです。

## 構成

```
line-qlora/
├── chatlog/             # LINEのチャットログ（.txt）を置くフォルダ
├── corpus/              # 生成された学習コーパス（.jsonl）
├── parse_chatlog.py     # チャットログのパース
├── select_speaker.py    # 模倣する話者の選択
├── build_corpus.py      # 学習用コーパスの生成
├── train.py             # QLoRAによるファインチューニング
├── inference.py         # 学習済みモデルで対話
└── pyproject.toml       # Poetry設定
```

## 環境構築

### 前提条件

- Python 3.12
- NVIDIA GPU（VRAM 12GB以上推奨）
- Poetry

### インストール

```bash
poetry install
```

> PyTorchはCUDA 12.4対応版が自動的にインストールされます。

---

## 使い方

### 1. チャットログを用意する

LINEアプリからトーク履歴をエクスポートし、`chatlog/` フォルダに `.txt` ファイルとして置きます。

**エクスポート手順（iOS / Android 共通）**
1. トーク画面右上のメニュー →「トーク設定」
2. 「トークのバックアップ」→「テキストで送る」
3. 送られてきた `.txt` ファイルを `chatlog/` に配置

複数人のトークを追加するほどコーパスが豊富になり、精度が上がります。

---

### 2. チャットログをパースする（確認用）

```bash
poetry run python parse_chatlog.py chatlog/<ファイル名>.txt
```

**出力例**
```
話者: ['自分', '友人A']
メッセージ数: 500
[2025.12.30 火曜日] 23:47 友人A: おつかれ！
...
```

---

### 3. コーパスを生成する

```bash
poetry run python build_corpus.py
```

実行すると対話形式でチャットログと話者を選択できます。選択後、`chatlog/` 内の残りのファイルについても同じ人物を抽出するか順番に確認され、複数トークのデータをまとめて1つのコーパスに統合できます。

```
=== チャットログ一覧 ===
  [0] chat_A.txt
  [1] group_chat.txt
  ...

使用するログの番号を入力してください: 0

=== 話者一覧 ===
  [0] 自分
  [1] 友人A

模倣したい話者の番号を入力してください: 1

話者「友人A」を選択しました（発言数: 200件）

他のチャットログから「友人A」を追加で収集できます。

--- group_chat.txt ---
  このファイルから「友人A」を抽出しますか？ [y/n]: y
  → 「友人A」が見つかりました（発言数: 80件）

合計発言数: 280件（2ファイル分）
コーパスを保存しました: corpus/友人A_corpus.jsonl（120件）
```

**同じ人物が別の表示名で登録されている場合**（例：グループトークでは別名など）、話者一覧から手動でマッピングできます。

ファイルパスと話者名を直接指定することもできます。

```bash
poetry run python build_corpus.py \
  --chatlog chatlog/chat_A.txt \
  --speaker 友人A \
  --context-turns 3 \
  --max-gap-minutes 60
```

生成されるコーパスのフォーマット（JSONL）:

```json
{
  "messages": [
    {"role": "system",    "content": "あなたは友人Aです..."},
    {"role": "user",      "content": "最近どう？"},
    {"role": "assistant", "content": "まあまあかな〜"}
  ]
}
```

---

### 4. ファインチューニングする

```bash
poetry run python train.py --corpus corpus/友人A_corpus.jsonl
```

**主なオプション**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--corpus` | （必須） | コーパスJSONLのパス |
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | ベースモデルのHuggingFace ID |
| `--output-dir` | `./models` | モデルの保存先 |
| `--epochs` | `3` | 学習エポック数 |
| `--batch-size` | `2` | バッチサイズ |
| `--grad-accum` | `8` | 勾配蓄積ステップ数 |
| `--max-seq-length` | `1024` | 最大シーケンス長 |
| `--lr` | `2e-4` | 学習率 |

**3Bモデルで軽量に動かす例**

```bash
poetry run python train.py \
  --corpus corpus/友人A_corpus.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --epochs 5
```

学習が完了すると `models/` にLoRAアダプタが保存されます。

---

### 5. 対話する

```bash
poetry run python inference.py --speaker 友人A
```

```
--- 友人Aとの会話を開始します（終了: 'exit' または Ctrl+C）---

あなた: 最近どう？
友人A: まあまあかな〜 なんか眠いけど笑

あなた: exit
会話を終了します。
```

**主なオプション**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--speaker` | （必須） | 模倣する話者名 |
| `--model-dir` | `./models` | LoRAアダプタのディレクトリ |
| `--base-model` | `Qwen/Qwen2.5-7B-Instruct` | ベースモデルのHuggingFace ID |
| `--max-history` | `6` | 保持する会話ターン数 |

---

## VRAM目安

| モデル | 量子化 | 必要VRAM |
|---|---|---|
| Qwen2.5-3B-Instruct | 4bit QLoRA | 〜6GB |
| Qwen2.5-7B-Instruct | 4bit QLoRA | 〜10GB |
| Qwen2.5-14B-Instruct | 4bit QLoRA | 〜18GB |

---

## コーパスの質を上げるには

- **複数のトークを追加する**: `chatlog/` に複数の `.txt` を置いて、同じ話者のデータをまとめてコーパス化することで精度が上がります
- **データ量の目安**: 最低50〜100サンプル以上あると効果が出やすいです
- **コンテキストターン数**: `--context-turns` を増やすと文脈を長く保持しますが、サンプル数が減ります
- **セッション分割（`--max-gap-minutes`）**: 時間的に離れたメッセージが誤って「刺激→反応」ペアに混入しないよう、一定時間以上の空白で会話を分割します。デフォルトは60分
