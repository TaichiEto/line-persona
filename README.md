LINEのトーク履歴をコーパスにして、特定の人物の口調を模倣するLLMをファインチューニングするプロジェクトです。

## 構成

```
line-persona/
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
話者: ['衛藤泰地', 'natsumi_']
メッセージ数: 29
[2025.12.30 火曜日] 23:47 natsumi_: 風邪引いてしまったのでずっと家にいます
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
  [0] [LINE]natsumi_.txt
  [1] [LINE]group_chat.txt
  ...

使用するログの番号を入力してください: 0

=== 話者一覧 ===
  [0] 衛藤泰地
  [1] natsumi_

模倣したい話者の番号を入力してください: 1

話者「natsumi_」を選択しました（発言数: 10件）

他のチャットログから「natsumi_」を追加で収集できます。

--- [LINE]group_chat.txt ---
  このファイルから「natsumi_」を抽出しますか？ [y/n]: y
  → 「natsumi_」が見つかりました（発言数: 15件）

合計発言数: 25件（2ファイル分）
コーパスを保存しました: corpus/natsumi__corpus.jsonl（12件）
```

**同じ人物が別の表示名で登録されている場合**（例：グループトークでは「なつみ」など）、話者一覧から手動でマッピングできます。Enterを押すとそのファイルをスキップします。

```
--- [LINE]group_chat.txt ---
  このファイルから「natsumi_」を抽出しますか？ [y/n]: y
  → 「natsumi_」はこのファイルに見つかりませんでした。
  別の名前で登場しているかもしれません。

=== 話者一覧（Enterでスキップ）===
  [0] なつみ
  [1] 田中太郎

話者の番号を入力してください（スキップ: Enter）: 0
  → 「なつみ」を「natsumi_」として追加しました（発言数: 8件）
```

ファイルパスと話者名を直接指定することもできます。

```bash
poetry run python build_corpus.py \
  --chatlog chatlog/[LINE]natsumi_.txt \
  --speaker natsumi_ \
  --context-turns 3        # 直前の会話ターン数（デフォルト: 3）
  --max-gap-minutes 60     # セッション分割の時間ギャップ閾値・分（デフォルト: 60）
```

生成されるコーパスのフォーマット（JSONL）:

```json
{
  "messages": [
    {"role": "system",    "content": "あなたはnatsumi_です..."},
    {"role": "user",      "content": "あら大丈夫？"},
    {"role": "assistant", "content": "授業始まるまでにはなおりそうなので大丈夫です！"}
  ]
}
```

---

### 4. ファインチューニングする

```bash
poetry run python train.py --corpus corpus/natsumi__corpus.jsonl
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

**別のモデルを使う例（3Bモデルで軽量に）**

```bash
poetry run python train.py \
  --corpus corpus/natsumi__corpus.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --epochs 5
```

学習が完了すると `models/` にLoRAアダプタが保存されます。

---

### 5. 対話する

```bash
poetry run python inference.py --speaker natsumi_
```

```
--- natsumi_との会話を開始します（終了: 'exit' または Ctrl+C）---

あなた: 最近どう？
natsumi_: 風邪引いてたんですけどなおりました！

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
- **セッション分割（`--max-gap-minutes`）**: 時間的に離れたメッセージが誤って「刺激→反応」ペアに混入しないよう、一定時間以上の空白で会話を分割します。デフォルトは60分。LINEの会話が飛び飛びになりがちな相手には短く（例: `30`）、長いやり取りが多い場合は長く設定すると精度が上がります
