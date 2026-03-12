"""
parse_chatlog.py

LINEのチャットログ（.txt）をパースして構造化データに変換するモジュール。

対応フォーマット:
  [A] 日本語エクスポート形式
      2025.12.30 火曜日
      23:43 natsumi_ スタンプ
      23:47 衛藤泰地 わ！
      見ます!      ← タイムスタンプなし（継続行）

  [B] 英語エクスポート形式
      [LINE] Chat history with 美聡 Misato
      Saved on: 2026/03/08, 22:10

      2026/01/25 Sun
      07:53	美聡 Misato	[Sticker]
      20:07	衛藤泰地	"去年の春とった！！！
      いいよね霧笑笑"      ← "..." で囲まれた複数行メッセージ
"""

import re
from dataclasses import dataclass
from pathlib import Path

# ---- スキップキーワード ----
SKIP_KEYWORDS = [
    # 日本語
    "スタンプ", "写真", "動画", "ファイル", "GIF",
    "通話", "音声通話", "ビデオ通話",
    "メッセージの送信を取り消しました",
    "LINE Keep",
    # 英語
    "[Sticker]", "[Photo]", "[Video]", "[File]", "[GIF]",
    "[Voice message]", "[Audio]",
    "You unsent a message.", "You unsent a photo.",
    "Call", "Video call",
]

# ---- フォーマットA パターン ----
_DATE_A = re.compile(r"^\d{4}\.\d{2}\.\d{2}\s+\S+曜日$")
_MSG_A  = re.compile(r"^(\d{2}:\d{2}) (\S+) (.*)$")

# ---- フォーマットB パターン ----
_DATE_B = re.compile(r"^\d{4}/\d{2}/\d{2}\s+\w+$")
_MSG_B  = re.compile(r"^(\d{2}:\d{2})\t(.+)\t(.*)$")
_SYS_B  = re.compile(r"^\d{2}:\d{2}\t\t")   # 空スピーカー（システムメッセージ）


@dataclass
class Message:
    date: str
    time: str
    speaker: str
    content: str


def _should_skip(content: str) -> bool:
    return any(kw in content for kw in SKIP_KEYWORDS)


def detect_format(filepath: str | Path) -> str:
    """
    チャットログのフォーマットを自動検出する。

    Returns
    -------
    "A" : 日本語エクスポート形式
    "B" : 英語エクスポート形式
    """
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[LINE] Chat history"):
                return "B"
            if _DATE_A.match(line):
                return "A"
            if _DATE_B.match(line):
                return "B"
    return "A"


def _parse_format_a(filepath: Path) -> list[Message]:
    """日本語エクスポート形式をパースする。"""
    messages: list[Message] = []
    current_date = ""
    current_msg: Message | None = None

    with open(filepath, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if _DATE_A.match(line):
                current_date = line
                current_msg = None
                continue

            m = _MSG_A.match(line)
            if m:
                time, speaker, content = m.group(1), m.group(2), m.group(3)
                if _should_skip(content):
                    current_msg = None
                    continue
                current_msg = Message(date=current_date, time=time,
                                      speaker=speaker, content=content)
                messages.append(current_msg)
                continue

            # 継続行
            stripped = line.strip()
            if stripped and current_msg is not None:
                current_msg.content += "\n" + stripped

    return messages


def _parse_format_b(filepath: Path) -> list[Message]:
    """英語エクスポート形式をパースする。"""
    messages: list[Message] = []
    current_date = ""
    current_msg: Message | None = None
    in_quote = False          # "..." ブロック内フラグ

    with open(filepath, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # --- 引用ブロックの継続処理 ---
            if in_quote:
                if current_msg is not None:
                    current_msg.content += "\n" + line.lstrip('"')
                # 行末が " で終わったら引用ブロック終了
                if line.rstrip().endswith('"'):
                    in_quote = False
                    if current_msg is not None:
                        # 前後の引用符を除去
                        current_msg.content = current_msg.content.strip('"').strip()
                continue

            # --- ヘッダー・空行スキップ ---
            if (line.startswith("[LINE] Chat history")
                    or line.startswith("Saved on:")
                    or line.strip() == ""):
                current_msg = None
                continue

            # --- 日付行 ---
            if _DATE_B.match(line):
                current_date = line.strip()
                current_msg = None
                continue

            # --- システムメッセージ（空スピーカー、またはタイムスタンプのみ行） ---
            if _SYS_B.match(line):
                current_msg = None
                continue

            # --- 通常メッセージ行 ---
            m = _MSG_B.match(line)
            if m:
                time, speaker, content = m.group(1), m.group(2), m.group(3)
                if _should_skip(content):
                    current_msg = None
                    continue

                # "..." ブロック開始チェック
                if content.startswith('"') and not content.rstrip().endswith('"'):
                    in_quote = True
                    content = content.lstrip('"')

                current_msg = Message(date=current_date, time=time,
                                      speaker=speaker, content=content)
                messages.append(current_msg)
                continue

            # --- 継続行（フォーマットBでも稀に発生） ---
            stripped = line.strip()
            if stripped and current_msg is not None and not in_quote:
                current_msg.content += "\n" + stripped

    return messages


def parse_file(filepath: str | Path) -> list[Message]:
    """
    LINEチャットログファイルをパースしてMessageリストを返す。

    フォーマット（A: 日本語 / B: 英語）は自動検出する。

    Parameters
    ----------
    filepath : str | Path
        チャットログのパス（.txt）

    Returns
    -------
    list[Message]
        パース済みメッセージ一覧
    """
    filepath = Path(filepath)
    fmt = detect_format(filepath)
    if fmt == "B":
        return _parse_format_b(filepath)
    return _parse_format_a(filepath)


def get_speakers(messages: list[Message]) -> list[str]:
    """メッセージ一覧から話者名の一覧（登場順）を返す。"""
    seen: dict[str, None] = {}
    for msg in messages:
        seen[msg.speaker] = None
    return list(seen.keys())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_chatlog.py <chatlog.txt>")
        sys.exit(1)

    msgs = parse_file(sys.argv[1])
    speakers = get_speakers(msgs)
    fmt = detect_format(sys.argv[1])
    print(f"フォーマット: {'日本語' if fmt == 'A' else '英語'}")
    print(f"話者: {speakers}")
    print(f"メッセージ数: {len(msgs)}")
    for msg in msgs[:10]:
        content_preview = msg.content.replace("\n", " / ")[:60]
        print(f"[{msg.date}] {msg.time} {msg.speaker}: {content_preview}")
