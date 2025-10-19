"""Filter out judge errors from assistantturns_scored.jsonl"""
import json

input_file = "assistantturns_scored.jsonl"
clean_file = "assistantturns_scored_clean.jsonl"
errors_file = "assistantturns_errors.jsonl"

valid = 0
errors = 0
total_input = 0

# Judge fields to remove for retry
judge_fields = ['judge_safe', 'judge_reason', 'judge_error', 'judge_provider']

# Required fields that should exist in all records
required_fields = ['assistant_turn_id', 'user_msg', 'assistant_msg']

with open(input_file, 'r', encoding='utf-8') as inf, \
     open(clean_file, 'w', encoding='utf-8') as cleanf, \
     open(errors_file, 'w', encoding='utf-8') as errf:

    for line in inf:
        total_input += 1
        data = json.loads(line)

        # Validate required fields exist
        for field in required_fields:
            assert field in data, f"Missing required field '{field}' in line {total_input}"

        if 'judge_error' in data:
            # Validate it actually has judge_error
            assert data['judge_error'] is not None, f"Null judge_error in line {total_input}"

            # Remove judge fields for retry
            for field in judge_fields:
                data.pop(field, None)

            # Validate judge fields are removed
            for field in judge_fields:
                assert field not in data, f"Failed to remove {field} in line {total_input}"

            errf.write(json.dumps(data, ensure_ascii=False) + '\n')
            errors += 1
        else:
            # Validate it has valid judge fields
            assert 'judge_safe' in data, f"Missing judge_safe in valid record line {total_input}"
            assert data['judge_safe'] in [0, 1], f"Invalid judge_safe value in line {total_input}"

            cleanf.write(line)
            valid += 1

# Final validation
assert total_input == valid + errors, f"Count mismatch: {total_input} != {valid} + {errors}"

# Verify output files
with open(clean_file, 'r', encoding='utf-8') as f:
    clean_count = sum(1 for _ in f)
assert clean_count == valid, f"Clean file has {clean_count} rows, expected {valid}"

with open(errors_file, 'r', encoding='utf-8') as f:
    error_count = sum(1 for _ in f)
assert error_count == errors, f"Error file has {error_count} rows, expected {errors}"

print(f"[OK] Validation passed")
print(f"  Total input: {total_input}")
print(f"  Valid: {valid}")
print(f"  Errors: {errors}")
print(f"  Clean data: {clean_file}")
print(f"  Errors (ready for retry): {errors_file}")
