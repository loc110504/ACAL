import unittest
import sys
import json
import csv
from datetime import datetime
from state import Argument, GraphState
from node import (
    rag_retrieval,
    overall_options,
    agent_selector,
    multi_agent_argument_generator,
    human_review,
    route_after_human_review,
    argument_validator,
    final_answer_generator
)

TSV_PATH = "data/learned_hands_courts/test.tsv"


class TestFullLegalWorkflow(unittest.TestCase):

    def run_single_sample(self, task_info: str, sample_index: int):
        """
        Run full workflow for one sample and return serializable result
        """
        state = {
            "task_name": "learned hands courts",
            "task_info": task_info,
            "enable_streaming": False
        }

        # === Workflow ===
        state = rag_retrieval(state)
        state = overall_options(state)
        state = agent_selector(state, "support")
        state = agent_selector(state, "attack")
        state = multi_agent_argument_generator(state)

        state = human_review(state)
        state["human_review_complete"] = True
        route_after_human_review(state)

        state = argument_validator(state)
        state = final_answer_generator(state)

        # === Serialize result ===
        result = {
            "sample_index": sample_index,
            "task_name": state.get("task_name"),
            "task_info": state.get("task_info"),
            "final_answer": state.get("final_answer", {}),
            "options": state.get("options", []),
            "retrieved_documents_count": len(state.get("retrieved_documents", [])),
            "arguments": [],
            "validated_arguments": []
        }

        for arg in state.get("arguments", []):
            result["arguments"].append({
                "content": arg.content,
                "type": arg.argument_type,
                "parent_option": arg.parent_option,
                "agent_role": getattr(arg, "agent_role", None),
                "agent_name": getattr(arg, "agent_name", None)
            })

        for arg in state.get("validated_arguments", []):
            result["validated_arguments"].append({
                "content": arg.content,
                "type": arg.argument_type,
                "parent_option": arg.parent_option,
                "validity_score": getattr(arg, "validity_score", None),
                "agent_role": getattr(arg, "agent_role", None),
                "agent_name": getattr(arg, "agent_name", None),
                "supporting_docs": getattr(arg, "supporting_docs", [])
            })

        return result

    def test_inference_from_tsv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"workflow_results_{timestamp}.json"
        
        # Initialize the JSON file with opening bracket
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("[\n")
        
        first_result = True
        total_processed = 0
        
        with open(TSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)  # Read all rows to get total count
            total_rows = len(rows)
            
            for row in rows:
                sample_index = int(row["index"])
                task_info = row["text"]

                print(f"\n{'='*60}")
                print(f"Running sample {total_processed + 1}/{total_rows} (index: {sample_index})")
                print(f"{'='*60}")
                
                try:
                    result = self.run_single_sample(task_info, sample_index)
                    
                    # Append result to file immediately
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        if not first_result:
                            out_f.write(",\n")  # Add comma before next item
                        json.dump(result, out_f, indent=2, ensure_ascii=False)
                        out_f.flush()  # Ensure data is written to disk
                    
                    first_result = False
                    total_processed += 1
                    print(f"\n✅ Sample {sample_index} completed and saved to {output_file}")
                    
                except Exception as e:
                    print(f"\n❌ Error processing sample {sample_index}: {e}")
                    # Log error but continue with next sample
                    error_result = {
                        "sample_index": sample_index,
                        "error": str(e),
                        "task_info": task_info[:200] + "..." if len(task_info) > 200 else task_info
                    }
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        if not first_result:
                            out_f.write(",\n")
                        json.dump(error_result, out_f, indent=2, ensure_ascii=False)
                        out_f.flush()
                    first_result = False
                    total_processed += 1
        
        # Close the JSON array
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n]")
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETED: Processed {total_processed}/{total_rows} samples")
        print(f"📁 Results saved to: {output_file}")
        print(f"{'='*60}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
