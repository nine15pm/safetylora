"""
Filter unsafe responses from scored assistant turns.
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "assistantturns_scored.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "assistantturns_unsafe.jsonl"


def main():
    safe_count = 0
    unsafe_count = 0
    error_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            try:
                entry = json.loads(line.strip())

                if entry["judge_safe"] == 0:
                    unsafe_count += 1
                    outfile.write(line)
                elif entry["judge_safe"] == 1:
                    safe_count += 1
                else:
                    error_count += 1

            except (json.JSONDecodeError, KeyError) as e:
                error_count += 1

    # Print stats
    total = safe_count + unsafe_count + error_count
    print(f"\nFiltering Results:")
    print(f"  Total entries: {total}")
    print(f"  Safe: {safe_count} ({100*safe_count/total:.1f}%)")
    print(f"  Unsafe: {unsafe_count} ({100*unsafe_count/total:.1f}%)")
    print(f"  Errors: {error_count}")
    print(f"\nWrote {unsafe_count} unsafe entries to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
