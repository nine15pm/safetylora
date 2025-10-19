"""Assemble final training dataset from safe and rewritten assistant turns."""

import json
import uuid
from pathlib import Path


def assemble_final_dataset():
    """Combine safe and rewritten records into final dataset format."""
    datagen_dir = Path(__file__).parent

    # Read userturns to get prompt_type mapping
    print("Reading userturns for prompt_type mapping...")
    user_turn_prompt_types = {}
    with open(datagen_dir / "userturns.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            user_turn = json.loads(line.strip())
            # Map user_turn_id to type field (converted to lowercase)
            user_turn_prompt_types[user_turn["user_turn_id"]] = user_turn["type"].lower()

    print(f"Loaded prompt_type for {len(user_turn_prompt_types)} user turns")

    final_records = []

    # Process safe records
    print("\nProcessing safe records...")
    safe_count = 0
    with open(datagen_dir / "assistantturns_scored_safe.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())

            final_record = {
                "id": str(uuid.uuid4()),
                "category": record["category"],
                "prompt_type": user_turn_prompt_types[record["user_turn_id"]],
                "tag_sp": "base" if record["system_prompt_id"] == "sp1" else "variant",
                "seed_id": record["seed_id"],
                "sp_id": record["system_prompt_id"],
                "user_turn_id": record["user_turn_id"],
                "user_msg": record["user_msg"],
                "assistant_turn_id": record["assistant_turn_id"],
                "assistant_msg": record["assistant_msg"],
            }
            final_records.append(final_record)
            safe_count += 1

    print(f"Processed {safe_count} safe records")

    # Process rewritten records (use rewritten_msg as assistant_msg)
    print("\nProcessing rewritten records...")
    rewritten_count = 0
    with open(datagen_dir / "assistantturns_rewritten.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())

            final_record = {
                "id": str(uuid.uuid4()),
                "category": record["category"],
                "prompt_type": user_turn_prompt_types[record["user_turn_id"]],
                "tag_sp": "base" if record["system_prompt_id"] == "sp1" else "variant",
                "seed_id": record["seed_id"],
                "sp_id": record["system_prompt_id"],
                "user_turn_id": record["user_turn_id"],
                "user_msg": record["user_msg"],
                "assistant_turn_id": record["assistant_turn_id"],
                "assistant_msg": record["rewritten_msg"],  # Use rewritten message
            }
            final_records.append(final_record)
            rewritten_count += 1

    print(f"Processed {rewritten_count} rewritten records")

    # Write final dataset
    output_path = datagen_dir / "dataset.jsonl"
    print(f"\nWriting final dataset to {output_path.name}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in final_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nFinal dataset summary:")
    print(f"  Safe records: {safe_count}")
    print(f"  Rewritten records: {rewritten_count}")
    print(f"  Total records: {len(final_records)}")
    print(f"\nCreated: {output_path}")


if __name__ == "__main__":
    assemble_final_dataset()
