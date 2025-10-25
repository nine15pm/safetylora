"""
Prepare dataset for training by adding blank system prompt variants and enriching with prompt text.
"""
import json
import uuid
from pathlib import Path


def load_sysprompts(path):
    """Load system prompts and create sp_id -> prompt text lookup."""
    sp_lookup = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sp = json.loads(line)
            sp_lookup[sp['id']] = sp['prompt']
    return sp_lookup


def load_dataset(path):
    """Load dataset as list of dictionaries."""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def create_blank_duplicate(row):
    """Create a duplicate of a row with blank system prompt."""
    duplicate = row.copy()
    duplicate['id'] = str(uuid.uuid4())
    duplicate['tag_sp'] = 'blank'
    duplicate['sp_id'] = 'blank'
    return duplicate


def add_system_prompt_text(row, sp_lookup):
    """Add system prompt text to row based on sp_id."""
    sp_id = row['sp_id']
    if sp_id == 'blank':
        row['sp'] = ''
    else:
        row['sp'] = sp_lookup.get(sp_id, '')
    return row


def save_dataset(dataset, path):
    """Save dataset as JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # Load system prompts
    sp_lookup = load_sysprompts('data/sysprompts.jsonl')
    print(f"Loaded {len(sp_lookup)} system prompts")

    # Load dataset
    dataset = load_dataset('data/dataset.jsonl')
    print(f"Loaded {len(dataset)} rows")

    # Create blank duplicates for base rows
    base_rows = [row for row in dataset if row['tag_sp'] == 'base']
    blank_rows = [create_blank_duplicate(row) for row in base_rows]
    print(f"Created {len(blank_rows)} blank duplicates from base rows")

    # Combine original + duplicates
    dataset_expanded = dataset + blank_rows
    print(f"Total rows after expansion: {len(dataset_expanded)}")

    # Add system prompt text to all rows
    dataset_enriched = [add_system_prompt_text(row, sp_lookup) for row in dataset_expanded]
    print(f"Added system prompt text to all rows")

    # Count tag_sp categories
    tag_counts = {}
    for row in dataset_enriched:
        tag = row['tag_sp']
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"tag_sp distribution: {tag_counts}")

    # Save enriched dataset
    output_path = 'data/dataset.jsonl'
    save_dataset(dataset_enriched, output_path)
    print(f"\nSaved {len(dataset_enriched)} rows to {output_path}")

    # Verify samples
    print("\nVerification samples:")
    sample_blank = next(row for row in dataset_enriched if row['tag_sp'] == 'blank')
    print(f"  Blank row - sp_id: '{sample_blank['sp_id']}', sp length: {len(sample_blank['sp'])}")

    sample_base = next(row for row in dataset_enriched if row['tag_sp'] == 'base')
    print(f"  Base row - sp_id: '{sample_base['sp_id']}', sp length: {len(sample_base['sp'])}")

    sample_variant = next(row for row in dataset_enriched if row['tag_sp'] == 'variant')
    print(f"  Variant row - sp_id: '{sample_variant['sp_id']}', sp length: {len(sample_variant['sp'])}")
