"""
select_speaker.py

chatlogフォルダにあるLINEチャットログから
模倣したい話者をCLIで選択するモジュール。

複数ファイルにまたがる同一人物のメッセージも収集できる。
"""

from pathlib import Path

from parse_chatlog import Message, get_speakers, parse_file

CHATLOG_DIR = Path(__file__).parent / "chatlog"


def list_chatlog_files() -> list[Path]:
    """chatlogフォルダ内の .txt ファイルを一覧で返す。"""
    return sorted(CHATLOG_DIR.glob("*.txt"))


def _ask_yes_no(prompt: str) -> bool:
    """y/n の入力を受け付ける。"""
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("  ※ y か n を入力してください")


def select_chatlog(files: list[Path]) -> Path:
    """CLIでチャットログファイルを選択させる。"""
    if not files:
        raise FileNotFoundError(f"chatlogフォルダにtxtファイルがありません: {CHATLOG_DIR}")

    print("\n=== チャットログ一覧 ===")
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}")

    while True:
        raw = input("\n使用するログの番号を入力してください: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(files):
            return files[int(raw)]
        print("  ※ 有効な番号を入力してください")


def select_speaker(speakers: list[str]) -> str:
    """CLIで模倣する話者を選択させる。"""
    print("\n=== 話者一覧 ===")
    for i, name in enumerate(speakers):
        print(f"  [{i}] {name}")

    while True:
        raw = input("\n模倣したい話者の番号を入力してください: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(speakers):
            return speakers[int(raw)]
        print("  ※ 有効な番号を入力してください")


def _select_speaker_optional(speakers: list[str]) -> str | None:
    """
    話者を選択させる（Enterでスキップ可）。

    Returns
    -------
    str | None
        選択した話者名、スキップした場合は None
    """
    print("\n=== 話者一覧（Enterでスキップ）===")
    for i, name in enumerate(speakers):
        print(f"  [{i}] {name}")

    while True:
        raw = input("\n話者の番号を入力してください（スキップ: Enter）: ").strip()
        if raw == "":
            return None
        if raw.isdigit() and 0 <= int(raw) < len(speakers):
            return speakers[int(raw)]
        print("  ※ 有効な番号を入力するか、Enterを押してください")


def collect_all_messages(
    speaker: str,
    initial_file: Path,
    initial_messages: list[Message],
    all_files: list[Path],
) -> list[Message]:
    """
    最初に選択したファイルに加え、他のチャットログからも
    同一人物のメッセージを収集して結合する。

    同じ名前が見つからない場合は別名を手動で選択できる。
    スキップも可能。

    Parameters
    ----------
    speaker : str
        最初に選択した話者名
    initial_file : Path
        最初に選択したチャットログファイル
    initial_messages : list[Message]
        最初のファイルのパース済みメッセージ
    all_files : list[Path]
        chatlogフォルダ内の全ファイル

    Returns
    -------
    list[Message]
        全ファイルから収集した統合メッセージリスト
        （speaker フィールドはすべて initial_speaker に正規化済み）
    """
    combined = list(initial_messages)

    initial_file = Path(initial_file).resolve()
    remaining = [f for f in all_files if f.resolve() != initial_file]
    if not remaining:
        return combined

    print(f"\n他のチャットログから「{speaker}」を追加で収集できます。")

    for path in remaining:
        print(f"\n--- {path.name} ---")
        if not _ask_yes_no(f"  このファイルから「{speaker}」を抽出しますか？"):
            continue

        messages = parse_file(path)
        speakers_in_file = get_speakers(messages)
        target_count = sum(1 for m in messages if m.speaker == speaker)

        if target_count > 0:
            # 同名の話者が見つかった
            print(f"  → 「{speaker}」が見つかりました（発言数: {target_count}件）")
            combined.extend(messages)
        else:
            # 同名が見つからない → 別名の可能性を確認
            print(f"  → 「{speaker}」はこのファイルに見つかりませんでした。")
            if speakers_in_file:
                print("  別の名前で登場しているかもしれません。")
                alias = _select_speaker_optional(speakers_in_file)
                if alias is not None:
                    # speaker名をaliasで置き換えてから統合
                    for m in messages:
                        if m.speaker == alias:
                            m.speaker = speaker  # 名前を正規化
                    alias_count = sum(1 for m in messages if m.speaker == speaker)
                    print(f"  → 「{alias}」を「{speaker}」として追加しました（発言数: {alias_count}件）")
                    combined.extend(messages)
                else:
                    print("  → スキップしました")
            else:
                print("  → 話者が見つからないためスキップしました")

    total = sum(1 for m in combined if m.speaker == speaker)
    print(f"\n合計発言数: {total}件（{len([f for f in all_files if f == initial_file or f in remaining])}ファイル分）")
    return combined


def run() -> tuple[Path, str, list[Message]]:
    """
    対話的にチャットログ・話者を選択し、複数ファイルからメッセージを収集して返す。

    Returns
    -------
    (initial_chatlog_path, speaker_name, combined_messages)
    """
    all_files = list_chatlog_files()
    chatlog_path = select_chatlog(all_files)

    print(f"\nパース中: {chatlog_path.name} ...")
    messages = parse_file(chatlog_path)
    speakers = get_speakers(messages)

    speaker = select_speaker(speakers)
    target_count = sum(1 for m in messages if m.speaker == speaker)
    print(f"\n話者「{speaker}」を選択しました（発言数: {target_count}件）")

    combined = collect_all_messages(speaker, chatlog_path, messages, all_files)

    return chatlog_path, speaker, combined


if __name__ == "__main__":
    chatlog_path, speaker, messages = run()
    print(f"\nログ: {chatlog_path.name}")
    print(f"話者: {speaker}")
    print(f"総メッセージ数: {len(messages)}")
