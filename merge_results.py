"""
Merge multiple workflow results JSON files into a single file.
Handles duplicates by keeping the most recent successful result for each sample_index.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import re


def extract_timestamp(filename: str) -> datetime:
    """Extract timestamp from filename like workflow_results_20260119_012638.json"""
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    return datetime.min


def load_json_file(filepath: Path) -> List[Dict]:
    """Load a single JSON file and return results list"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"⚠️  Warning: {filepath.name} does not contain a list, skipping...")
                return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing {filepath.name}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error reading {filepath.name}: {e}")
        return []


def merge_workflow_results(
    input_files: List[Path], 
    output_file: Path,
    keep_latest: bool = True,
    keep_successful_only: bool = False
) -> None:
    """
    Merge multiple workflow results files based on sample_index.
    
    PRIMARY KEY: sample_index
    - Each result MUST have a unique sample_index
    - Duplicates are resolved by timestamp or success status
    - Final output is sorted by sample_index
    
    Args:
        input_files: List of input JSON file paths
        output_file: Output merged JSON file path
        keep_latest: If True, keep the latest version for duplicate sample_index
        keep_successful_only: If True, only keep samples without errors
    """
    
    # Sort files by timestamp
    files_with_timestamps = [(f, extract_timestamp(f.name)) for f in input_files]
    files_with_timestamps.sort(key=lambda x: x[1])
    
    print(f"\n{'='*60}")
    print(f"MERGING {len(input_files)} WORKFLOW RESULT FILES")
    print(f"PRIMARY KEY: sample_index")
    print(f"{'='*60}\n")
    
    # Dictionary to store results: sample_index -> (result, timestamp, filename)
    # The sample_index is the UNIQUE IDENTIFIER for merging
    merged_results = {}
    
    total_loaded = 0
    total_errors = 0
    total_success = 0
    invalid_samples = 0
    
    for filepath, timestamp in files_with_timestamps:
        print(f"📂 Loading: {filepath.name}")
        results = load_json_file(filepath)
        
        if not results:
            print(f"   ⚠️  No results found\n")
            continue
        
        file_success = 0
        file_errors = 0
        file_duplicates = 0
        
        for result in results:
            # Validate sample_index exists and is valid
            if "sample_index" not in result:
                print(f"   ⚠️  Warning: Result without sample_index found, skipping")
                invalid_samples += 1
                continue
            
            sample_idx = result["sample_index"]
            
            # Validate sample_index is an integer
            if not isinstance(sample_idx, int):
                try:
                    sample_idx = int(sample_idx)
                    result["sample_index"] = sample_idx  # Normalize to int
                except (ValueError, TypeError):
                    print(f"   ⚠️  Warning: Invalid sample_index '{sample_idx}', skipping")
                    invalid_samples += 1
                    continue
            
            has_error = "error" in result
            
            if has_error:
                file_errors += 1
            else:
                file_success += 1
            
            # Skip if we only want successful results and this has an error
            if keep_successful_only and has_error:
                continue
            
            # Check if we already have this sample_index
            if sample_idx in merged_results:
                file_duplicates += 1
                existing_result, existing_ts, existing_file = merged_results[sample_idx]
                
                # Duplicate resolution strategy
                if keep_latest:
                    # Keep the most recent version based on timestamp
                    if timestamp > existing_ts:
                        merged_results[sample_idx] = (result, timestamp, filepath.name)
                        print(f"   🔄 Duplicate sample_index {sample_idx}: keeping newer version from {filepath.name}")
                else:
                    # Keep success over error regardless of timestamp
                    existing_has_error = "error" in existing_result
                    if existing_has_error and not has_error:
                        merged_results[sample_idx] = (result, timestamp, filepath.name)
                        print(f"   🔄 Duplicate sample_index {sample_idx}: replacing error with success")
            else:
                merged_results[sample_idx] = (result, timestamp, filepath.name)
        
        total_loaded += len(results)
        total_errors += file_errors
        total_success += file_success
        
        print(f"   ✅ Success: {file_success} | ❌ Errors: {file_errors} | 🔄 Duplicates: {file_duplicates}\n")
    
    # Extract results sorted by sample_index (PRIMARY KEY)
    final_results = [merged_results[idx][0] for idx in sorted(merged_results.keys())]
    
    # Count final statistics
    final_success = sum(1 for r in final_results if "error" not in r)
    final_errors = sum(1 for r in final_results if "error" in r)
    
    # Get sample_index range
    sample_indices = sorted(merged_results.keys())
    min_idx = min(sample_indices) if sample_indices else 0
    max_idx = max(sample_indices) if sample_indices else 0
    missing_indices = set(range(min_idx, max_idx + 1)) - set(sample_indices)
    
    # Save merged results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"{'='*60}")
    print(f"MERGE SUMMARY (by sample_index)")
    print(f"{'='*60}")
    print(f"📊 Total results loaded:    {total_loaded}")
    print(f"📊 Unique samples merged:   {len(final_results)}")
    print(f"📊 Sample index range:      {min_idx} to {max_idx}")
    if missing_indices:
        print(f"⚠️  Missing indices:        {len(missing_indices)} gaps (e.g., {sorted(list(missing_indices))[:5]}...)")
    print(f"✅ Successful samples:      {final_success}")
    print(f"❌ Failed samples:          {final_errors}")
    if invalid_samples > 0:
        print(f"⚠️  Invalid samples:        {invalid_samples} (missing/invalid sample_index)")
    print(f"🔄 Duplicates resolved:     {total_loaded - len(final_results) - invalid_samples}")
    print(f"\n💾 Merged results saved to: {output_file}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple workflow results JSON files into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge specific files
  python merge_results.py -i results1.json results2.json -o merged.json
  
  # Merge all workflow_results_*.json files in current directory
  python merge_results.py -p "workflow_results_*.json" -o merged_results.json
  
  # Merge and keep only successful results (no errors)
  python merge_results.py -p "workflow_results_*.json" -o merged.json --success-only
  
  # Merge but prefer earliest version of duplicates instead of latest
  python merge_results.py -p "workflow_results_*.json" -o merged.json --keep-earliest
        """
    )
    
    parser.add_argument(
        '-i', '--input-files',
        nargs='+',
        help='List of input JSON files to merge'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        help='Glob pattern to match input files (e.g., "workflow_results_*.json")'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output merged JSON file path'
    )
    
    parser.add_argument(
        '--success-only',
        action='store_true',
        help='Only keep successful results (exclude samples with errors)'
    )
    
    parser.add_argument(
        '--keep-earliest',
        action='store_true',
        help='Keep earliest version of duplicate samples (default: keep latest)'
    )
    
    args = parser.parse_args()
    
    # Collect input files
    input_files = []
    
    if args.input_files:
        input_files.extend([Path(f) for f in args.input_files])
    
    if args.pattern:
        pattern_files = list(Path('.').glob(args.pattern))
        input_files.extend(pattern_files)
    
    if not input_files:
        print("❌ Error: No input files specified. Use -i or -p to specify input files.")
        parser.print_help()
        return
    
    # Remove duplicates and filter existing files
    unique_files = []
    seen = set()
    for f in input_files:
        if f.resolve() not in seen and f.exists():
            unique_files.append(f)
            seen.add(f.resolve())
        elif not f.exists():
            print(f"⚠️  Warning: File not found: {f}")
    
    if not unique_files:
        print("❌ Error: No valid input files found.")
        return
    
    output_path = Path(args.output)
    
    # Merge the files
    merge_workflow_results(
        input_files=unique_files,
        output_file=output_path,
        keep_latest=not args.keep_earliest,
        keep_successful_only=args.success_only
    )


if __name__ == "__main__":
    main()
