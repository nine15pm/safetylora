"""
Convert intentseeds.md to structured JSONL format.
Each intent becomes a datapoint with category and type fields.
"""

import json
import re
from pathlib import Path


def standardize_category_name(category: str) -> str:
    """Convert category name to snake_case format."""
    # Remove parenthetical notes like "(SH/ED)" or "(Porn, Gambling, Paid Loot Boxes)"
    category = re.sub(r'\s*\([^)]+\)', '', category)
    # Replace special chars, hyphens, and spaces with underscores
    category = re.sub(r'[,&\s\-]+', '_', category)
    # Remove multiple underscores
    category = re.sub(r'_+', '_', category)
    # Remove leading/trailing underscores
    category = category.strip('_')
    # Convert to lowercase
    return category.lower()


def parse_intentseeds(input_path: Path, output_path: Path):
    """Parse intentseeds markdown and write to JSONL."""

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_category = None
    current_type = None
    datapoints = []

    for line in lines:
        line = line.rstrip()

        # Skip empty lines
        if not line:
            continue

        # Match top-level category (e.g., "## 1) Sexual & Romantic")
        category_match = re.match(r'^##\s+\d+\)\s+(.+)$', line)
        if category_match:
            current_category = standardize_category_name(category_match.group(1))
            continue

        # Match type (e.g., "### Adversarial (12)")
        type_match = re.match(r'^###\s+(\w+)\s+\(\d+\)$', line)
        if type_match:
            current_type = type_match.group(1)
            continue

        # If we have both category and type, and line doesn't start with #, it's an intent
        if current_category and current_type and not line.startswith('#'):
            intent = line.strip()
            if intent:  # Only add non-empty intents
                datapoints.append({
                    'category': current_category,
                    'type': current_type,
                    'intent': intent
                })

    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for dp in datapoints:
            f.write(json.dumps(dp) + '\n')

    print(f"Converted {len(datapoints)} intents to {output_path}")

    # Print summary
    category_counts = {}
    type_counts = {}
    for dp in datapoints:
        category_counts[dp['category']] = category_counts.get(dp['category'], 0) + 1
        type_counts[dp['type']] = type_counts.get(dp['type'], 0) + 1

    print("\nBy category:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")

    print("\nBy type:")
    for typ, count in type_counts.items():
        print(f"  {typ}: {count}")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    input_file = script_dir / 'intentseeds.md'
    output_file = script_dir / 'intentseeds.jsonl'

    parse_intentseeds(input_file, output_file)
