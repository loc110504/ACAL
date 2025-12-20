import json
import csv

JSON_PATH = "ours_hearsay.json"
GOLDEN_TSV_PATH = "data/hearsay/test.tsv"
OUTPUT_TSV_PATH = "ours_hearsay.tsv"


def load_golden_tsv(path):
    """
    Load golden TSV into dict by index
    """
    golden_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["index"])
            golden_map[idx] = {
                "golden_answer": row["answer"],
                "slice": row["slice"]
            }
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
    # Load files
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    golden_map = load_golden_tsv(GOLDEN_TSV_PATH)

    # Write TSV
    with open(OUTPUT_TSV_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "index",
            "text",
            "golden_answer",
            "final_answer",
            "slice",
            "explanation"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for item in results:
            idx = item["sample_index"]
            task_info = item.get("task_info", "")

            golden = golden_map.get(idx, {})
            golden_answer = golden.get("golden_answer", "")
            slice_ = golden.get("slice", "")

            final_answer_raw = item.get("final_answer", {}).get("answer", "")
            final_answer, explanation = parse_final_answer(final_answer_raw)

            writer.writerow({
                "index": idx,
                "text": task_info,
                "golden_answer": golden_answer,
                "final_answer": final_answer,
                "slice": slice_,
                "explanation": explanation
            })

    print(f"Saved TSV to: {OUTPUT_TSV_PATH}")


if __name__ == "__main__":
    main()
