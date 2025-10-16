"""
Convert safetypolicy.md to structured JSONL format.
Each category's policy becomes a datapoint with allowed/not_allowed/redirect fields.
"""

import json
import re
from pathlib import Path


def standardize_category_name(category: str) -> str:
    """Convert category name to snake_case format (matching intentseeds)."""
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


def parse_policy_section(lines, start_idx):
    """Parse a policy section starting from a category header."""
    allowed = []
    not_allowed = []
    redirect = []
    current_section = None

    i = start_idx + 1
    while i < len(lines):
        line = lines[i].rstrip()

        # Stop if we hit the next category header
        if line.startswith('### '):
            break

        # Check for section headers
        if line.startswith('**Allowed'):
            current_section = 'allowed'
            i += 1
            continue
        elif line.startswith('**Not Allowed'):
            current_section = 'not_allowed'
            i += 1
            continue
        elif line.startswith('**Redirect'):
            current_section = 'redirect'
            # Extract redirect content from the same line if present
            redirect_match = re.match(r'\*\*Redirect[^:]*:\*\*\s*(.+)', line)
            if redirect_match:
                redirect.append(redirect_match.group(1))
            i += 1
            continue

        # Collect content for current section
        if line and current_section:
            # Skip empty lines
            if line.strip():
                # Remove leading dashes for list items
                content = re.sub(r'^-\s*', '', line)
                if current_section == 'allowed':
                    allowed.append(content)
                elif current_section == 'not_allowed':
                    not_allowed.append(content)
                elif current_section == 'redirect':
                    redirect.append(content)

        i += 1

    return {
        'allowed': '\n'.join(allowed),
        'not_allowed': '\n'.join(not_allowed),
        'redirect': '\n'.join(redirect)
    }, i


def parse_safetypolicy(input_path: Path, output_path: Path):
    """Parse safetypolicy markdown and write to JSONL."""

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    policies = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Match category headers (e.g., "### 1) Sexual & Romantic")
        category_match = re.match(r'^###\s+\d+\)\s+(.+)$', line)
        if category_match:
            category_name = category_match.group(1)
            standardized_category = standardize_category_name(category_name)

            # Parse the policy sections for this category
            policy_data, next_idx = parse_policy_section(lines, i)

            policies.append({
                'category': standardized_category,
                'policy': policy_data
            })

            i = next_idx
        else:
            i += 1

    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for policy in policies:
            f.write(json.dumps(policy) + '\n')

    print(f"Converted {len(policies)} policy categories to {output_path}")

    # Print summary
    print("\nCategories:")
    for policy in policies:
        print(f"  {policy['category']}")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    input_file = script_dir / 'safetypolicy.md'
    output_file = script_dir / 'safetypolicy.jsonl'

    parse_safetypolicy(input_file, output_file)
