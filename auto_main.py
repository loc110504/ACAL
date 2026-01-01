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
        all_results = []

        with open(TSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sample_index = int(row["index"])
                task_info = row["text"]

                print(f"\nRunning sample index: {sample_index}")
                result = self.run_single_sample(task_info, sample_index)
                all_results.append(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"workflow_results_all_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nSaved all results to: {output_file}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
