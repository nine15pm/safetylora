"""
Data preparation: train/test splitting with stratification and weighted sampling.
"""
from datasets import load_dataset, concatenate_datasets, ClassLabel
from collections import defaultdict, Counter
import random


# Default sampling weights
DEFAULT_WEIGHTS = {
    'prompt_type': {
        'benign': 0.33,
        'borderline': 0.33,
        'adversarial': 0.34
    },
    'tag_sp': {
        'base': 0.50,
        'variant': 0.35,
        'blank': 0.25
    }
}


def split_train_test(dataset_path, test_size=0.2):
    """
    Split dataset into train/test with stratification on category and tag_sp.

    Args:
        dataset_path: Path to input jsonl file
        test_size: Fraction for test set (default 0.2)

    Returns:
        train_dataset, test_dataset
    """
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    # Create stratify column combining category and tag_sp
    def add_stratify_key(example):
        example['_stratify'] = f"{example['category']}_{example['tag_sp']}"
        return example

    dataset = dataset.map(add_stratify_key)

    # Convert stratify column to ClassLabel for stratification
    unique_keys = list(set(dataset['_stratify']))
    dataset = dataset.cast_column('_stratify', ClassLabel(names=unique_keys))

    # Split with stratification
    split = dataset.train_test_split(
        test_size=test_size,
        stratify_by_column='_stratify',
        seed=42
    )

    # Remove temporary stratify column
    train = split['train'].remove_columns(['_stratify'])
    test = split['test'].remove_columns(['_stratify'])

    return train, test


def apply_weighted_sampling(train_raw, weights=None):
    """
    Apply weighted sampling to train data while maintaining category balance.

    Args:
        train_raw: Raw training dataset
        weights: Dict with 'prompt_type' and 'tag_sp' weight configs

    Returns:
        Weighted training dataset
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    random.seed(42)

    # Group by category
    categories = set(train_raw['category'])
    print(f"\nApplying weighted sampling to {len(categories)} categories...")

    sampled_per_category = []

    for cat in sorted(categories):
        cat_data = train_raw.filter(lambda x: x['category'] == cat)

        # Group by (prompt_type, tag_sp)
        groups = defaultdict(list)
        for i, example in enumerate(cat_data):
            key = (example['prompt_type'], example['tag_sp'])
            groups[key].append(i)

        # Calculate target weights for each group (product of individual weights)
        target_weights = {}
        for pt in weights['prompt_type']:
            for tag in weights['tag_sp']:
                target_weights[(pt, tag)] = weights['prompt_type'][pt] * weights['tag_sp'][tag]

        # Find bottleneck: smallest (count / target_weight) ratio
        bottleneck_ratio = float('inf')
        for key, indices in groups.items():
            if target_weights[key] > 0:
                ratio = len(indices) / target_weights[key]
                if ratio < bottleneck_ratio:
                    bottleneck_ratio = ratio

        # Calculate target sample count for each group
        target_counts = {}
        for key in groups.keys():
            target_counts[key] = int(bottleneck_ratio * target_weights[key])

        # Sample from each group
        sampled_indices = []
        for key, indices in groups.items():
            n_target = target_counts[key]
            if n_target > 0:
                sampled = random.sample(indices, min(n_target, len(indices)))
                sampled_indices.extend(sampled)

        # Extract sampled data for this category
        cat_sampled = cat_data.select(sampled_indices)
        sampled_per_category.append(cat_sampled)

    # Combine all categories
    train_weighted = concatenate_datasets(sampled_per_category)

    return train_weighted


def verify_weights(dataset, weights):
    """Check final weighted distribution matches targets."""
    print("\n=== Weighted Distribution ===")

    pt_counts = Counter(dataset['prompt_type'])
    print(f"Prompt type (target):")
    for pt in sorted(weights['prompt_type'].keys()):
        actual_pct = 100 * pt_counts[pt] / len(dataset)
        target_pct = 100 * weights['prompt_type'][pt]
        print(f"  {pt}: {actual_pct:.1f}% (target {target_pct:.1f}%)")

    tag_counts = Counter(dataset['tag_sp'])
    print(f"\nTag SP (target):")
    for tag in sorted(weights['tag_sp'].keys()):
        actual_pct = 100 * tag_counts[tag] / len(dataset)
        target_pct = 100 * weights['tag_sp'][tag]
        print(f"  {tag}: {actual_pct:.1f}% (target {target_pct:.1f}%)")


def main():
    # Step 1: Split raw data
    train_raw, test = split_train_test('data/dataset.jsonl', test_size=0.2)

    print(f"Train raw: {len(train_raw)} samples")
    print(f"Test: {len(test)} samples")

    # Save raw splits
    train_raw.to_json('data/train_raw.jsonl')
    test.to_json('data/test_raw.jsonl')
    print("Saved raw splits")

    # Step 2: Apply weighted sampling to train
    train = apply_weighted_sampling(train_raw)
    print(f"\nFinal train: {len(train)} samples")

    verify_weights(train, DEFAULT_WEIGHTS)

    # Save final train set
    train.to_json('data/train.jsonl')
    print("\nSaved data/train.jsonl")


if __name__ == "__main__":
    main()
