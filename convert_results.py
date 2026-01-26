import json
import csv
import sys
import argparse

# Task configurations
TASK_CONFIGS = {
    "hearsay": {
        "json_path": "gemini_25flash_hearsay.json",
        "golden_tsv_path": "data/hearsay/test.tsv",
        "output_tsv_path": "gemini_25flash_hearsay.tsv",
        "has_slice": True  # Hearsay data has 'slice' column
    },
    "learned_hands_courts": {
        "json_path": "workflow_results_20260117_014605.json",
        "golden_tsv_path": "data/learned_hands_courts/test.tsv",
        "output_tsv_path": "learned_hands_courts_gemini25_flash_thinking.tsv",
        "has_slice": False  # Learned hands courts doesn't have 'slice' column
    }
}


def load_golden_tsv(path, has_slice=True):
    """
    Load golden TSV into dict by index
    """
    golden_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["index"])
            golden_data = {
                "golden_answer": row["answer"]
            }
            # Only include slice if it exists in the data
            if has_slice and "slice" in row:
                golden_data["slice"] = row["slice"]
            golden_map[idx] = golden_data
    return golden_map


def parse_final_answer(answer_str):
    """
    Parse stringified JSON inside final_answer["answer"]
    """
    try:
        parsed = json.loads(answer_str)
        return parsed.get("answer", ""), parsed.get("explanation", "")
    except Exception:
        return "", ""


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert JSON results to TSV format for evaluation"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["hearsay", "learned_hands_courts"],
        default="hearsay",
        help="Task type: hearsay or learned_hands_courts (default: hearsay)"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Custom path to JSON results file (overrides default)"
    )
    parser.add_argument(
        "--golden",
        type=str,
        help="Custom path to golden TSV file (overrides default)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom path to output TSV file (overrides default)"
    )
    
    args = parser.parse_args()
    
    # Get task configuration
    config = TASK_CONFIGS[args.task]
    
    # Use custom paths if provided, otherwise use defaults from config
    json_path = args.json if args.json else config["json_path"]
    golden_tsv_path = args.golden if args.golden else config["golden_tsv_path"]
    output_tsv_path = args.output if args.output else config["output_tsv_path"]
    has_slice = config["has_slice"]
    
    print(f"Processing task: {args.task}")
    print(f"  JSON input: {json_path}")
    print(f"  Golden TSV: {golden_tsv_path}")
    print(f"  Output TSV: {output_tsv_path}")
    print()
    
    # Load files
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    try:
        golden_map = load_golden_tsv(golden_tsv_path, has_slice=has_slice)
    except FileNotFoundError:
        print(f"Error: Golden TSV file not found: {golden_tsv_path}")
        sys.exit(1)

    # Write TSV
    with open(output_tsv_path, "w", encoding="utf-8", newline="") as f:
        # Build fieldnames based on whether task has slice column
        fieldnames = ["index", "text", "golden_answer", "final_answer"]
        if has_slice:
            fieldnames.append("slice")
        fieldnames.append("explanation")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for item in results:
            idx = item["sample_index"]
            task_info = item.get("task_info", "")

            golden = golden_map.get(idx, {})
            golden_answer = golden.get("golden_answer", "")

            final_answer_raw = item.get("final_answer", {}).get("answer", "")
            final_answer, explanation = parse_final_answer(final_answer_raw)

            row_data = {
                "index": idx,
                "text": task_info,
                "golden_answer": golden_answer,
                "final_answer": final_answer,
                "explanation": explanation
            }
            
            # Only add slice if this task has it
            if has_slice:
                row_data["slice"] = golden.get("slice", "")
            
            writer.writerow(row_data)

    print(f"✓ Saved TSV to: {output_tsv_path}")
    print(f"  Total samples processed: {len(results)}")


if __name__ == "__main__":
    main()
