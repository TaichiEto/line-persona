"""
build_corpus.py

パース済みLINEチャットログから、指定話者の発言を学習用コーパスに変換する。

出力形式（JSONL）:
  各行がJSONオブジェクトで、以下の形式:
  {
    "messages": [
      {"role": "system",    "content": "あなたは<話者名>です..."},
      {"role": "user",      "content": "<直前の相手の発言>"},
      {"role": "assistant", "content": "<話者の返答>"}
    ]
  }

複数ターンのコンテキストを持たせる場合は --context-turns で調整。
時間ギャップによるセッション分割は --max-gap-minutes で調整。
"""

import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from parse_chatlog import Message, parse_file
from select_speaker import collect_all_messages, list_chatlog_files, run as select_run

CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_DIR.mkdir(exist_ok=True)


def build_system_prompt(speaker: str) -> str:
    return (
        f"あなたは{speaker}です。"
        f"{speaker}の話し方・口調・言葉づかいを忠実に再現して返答してください。"
        "不自然に丁寧にならず、実際のLINEの会話のように自然に話してください。"
    )


def _parse_datetime(msg: Message) -> datetime | None:
    """メッセージの日時をdatetimeに変換する。"""
    date_str = msg.date
    time_str = msg.time
    try:
        if "曜日" in date_str:
            # Format A: "2025.12.30 火曜日"
            date_part = date_str.split()[0]
            return datetime.strptime(f"{date_part} {time_str}", "%Y.%m.%d %H:%M")
        else:
            # Format B: "2026/01/25 Sun"
            date_part = date_str.split()[0]
            return datetime.strptime(f"{date_part} {time_str}", "%Y/%m/%d %H:%M")
    except ValueError:
        return None


def _gap_minutes(msg1: Message, msg2: Message) -> float:
    """2メッセージ間の時間差（分）。解析不能なら0。"""
    dt1 = _parse_datetime(msg1)
    dt2 = _parse_datetime(msg2)
    if dt1 is None or dt2 is None:
        return 0.0
    return max(0.0, (dt2 - dt1).total_seconds() / 60)


def _split_into_sessions(
    messages: list[Message], max_gap_minutes: int
) -> list[list[Message]]:
    """
    時間ギャップでメッセージをセッション（会話の塊）に分割する。

    max_gap_minutes 以上の間隔があれば別セッションとみなす。
    会話の流れを壊さず、本当に「あるメッセージへの返答」だけを
    学習データとして使うために必要。
    """
    if not messages:
        return []

    sessions: list[list[Message]] = []
    current: list[Message] = [messages[0]]

    for prev, curr in zip(messages, messages[1:]):
        if _gap_minutes(prev, curr) >= max_gap_minutes:
            sessions.append(current)
            current = [curr]
        else:
            current.append(curr)

    sessions.append(current)
    return sessions


def _group_consecutive(messages: list[Message], speaker: str) -> list[dict]:
    """
    連続した発言をまとめて (role, content) のリストに変換する。

    同じ話者が連続して送ったメッセージは1つのターンにまとめる。
    """
    turns = []
    i = 0
    while i < len(messages):
        current_speaker = messages[i].speaker
        role = "assistant" if current_speaker == speaker else "user"
        contents = [messages[i].content]
        j = i + 1
        while j < len(messages) and messages[j].speaker == current_speaker:
            contents.append(messages[j].content)
            j += 1
        turns.append({"role": role, "content": "\n".join(contents)})
        i = j
    return turns


def _build_samples_from_session(
    session: list[Message],
    speaker: str,
    system_prompt: str,
    context_turns: int,
) -> list[dict]:
    """
    1セッション内のメッセージからサンプルを生成する。

    セッションをまたぐ会話は含まれないため、
    「このメッセージへの返答」という関係が保証される。
    """
    turns = _group_consecutive(session, speaker)
    samples = []

    for i, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue

        # context_turns分のコンテキストを取得（セッション内のみ）
        context = turns[max(0, i - context_turns) : i]
        if not context:
            continue

        # 直前のターンがuserでなければスキップ
        # （話者が連続して発言した場合、最初の1件だけ使う）
        if context[-1]["role"] != "user":
            continue

        chat_messages = [{"role": "system", "content": system_prompt}]
        chat_messages.extend(context)
        chat_messages.append({"role": "assistant", "content": turn["content"]})

        samples.append({"messages": chat_messages})

    return samples


def build_corpus(
    messages: list[Message],
    speaker: str,
    context_turns: int = 3,
    max_gap_minutes: int = 60,
) -> list[dict]:
    """
    メッセージ一覧から学習データ（辞書のリスト）を生成する。

    時間ギャップで会話をセッションに分割してから処理するため、
    「数時間前の無関係な発言に返答する」という誤ったサンプルを防ぐ。

    Parameters
    ----------
    messages : list[Message]
        パース済みメッセージ一覧
    speaker : str
        模倣したい話者名
    context_turns : int
        assistantの返答直前に含める会話ターン数（デフォルト3）
    max_gap_minutes : int
        セッション分割の時間ギャップ閾値・分（デフォルト60）

    Returns
    -------
    list[dict]
        学習データ（各要素がmessagesキーを持つ辞書）
    """
    system_prompt = build_system_prompt(speaker)
    sessions = _split_into_sessions(messages, max_gap_minutes)

    samples = []
    for session in sessions:
        samples.extend(
            _build_samples_from_session(session, speaker, system_prompt, context_turns)
        )

    return samples


def save_corpus(samples: list[dict], output_path: Path) -> None:
    """コーパスをJSONL形式で保存する。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"コーパスを保存しました: {output_path}（{len(samples)}件）")


def main() -> None:
    parser = ArgumentParser(description="LINEチャットログから学習コーパスを生成する")
    parser.add_argument("--chatlog", help="チャットログのパス（省略時は対話選択）")
    parser.add_argument("--speaker", help="模倣する話者名（省略時は対話選択）")
    parser.add_argument(
        "--context-turns",
        type=int,
        default=3,
        help="コンテキストとして含める会話ターン数（デフォルト: 3）",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=60,
        help="セッション分割の時間ギャップ閾値・分（デフォルト: 60）",
    )
    parser.add_argument("--output", help="出力JSONLのパス（省略時は自動生成）")
    args = parser.parse_args()

    if args.chatlog and args.speaker:
        chatlog_path = Path(args.chatlog)
        speaker = args.speaker
        messages = parse_file(chatlog_path)
        all_files = list_chatlog_files()
        messages = collect_all_messages(speaker, chatlog_path, messages, all_files)
    else:
        chatlog_path, speaker, messages = select_run()

    samples = build_corpus(
        messages,
        speaker,
        context_turns=args.context_turns,
        max_gap_minutes=args.max_gap_minutes,
    )

    if not samples:
        print("学習データが生成できませんでした。チャットログや話者名を確認してください。")
        sys.exit(1)

    output_path = (
        Path(args.output)
        if args.output
        else CORPUS_DIR / f"{speaker}_corpus.jsonl"
    )
    save_corpus(samples, output_path)

    # セッション統計
    from parse_chatlog import parse_file as _pf  # noqa: F401
    sessions = _split_into_sessions(messages, args.max_gap_minutes)
    print(f"\nセッション数: {len(sessions)}")
    print(f"サンプル数:   {len(samples)}")

    # サンプル表示
    print("\n--- サンプルデータ（先頭1件）---")
    sample = samples[0]
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"][:80].replace("\n", " ")
        print(f"  [{role}] {content}")


if __name__ == "__main__":
    main()
