from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from llm_caller import call_llm
import json
import re


@dataclass
class ArgumentRelation:
    """Represents a semantic relationship between two arguments"""
    source_id: str
    target_id: str
    relation_type: str  # "attack", "support", "neutral"
    confidence: float  # 0-1
    reasoning: str  # Explanation of why this relation exists


class SemanticRelationAnalyzer:
    def __init__(self, threshold_attack: float = 0.6, threshold_support: float = 0.6):
        """
        Args:
            threshold_attack: Minimum confidence to establish attack relation
            threshold_support: Minimum confidence to establish support relation
        """
        self.threshold_attack = threshold_attack
        self.threshold_support = threshold_support
        
    def analyze_pairwise_relation(
        self, 
        arg1_content: str, 
        arg1_option: str,
        arg1_type: str,
        arg2_content: str,
        arg2_option: str,
        arg2_type: str,
        task_context: str = ""
    ) -> ArgumentRelation:
        """
        Analyze the semantic relationship between two arguments using LLM-based NLI.
        
        Returns:
            ArgumentRelation with type (attack/support/neutral) and confidence
        """
        
        prompt = f"""You are an expert in legal argumentation and natural language inference.

Task: Analyze the semantic relationship between two legal arguments.

Context: {task_context if task_context else "Legal reasoning about hearsay evidence"}

Argument 1 (Supporting option: {arg1_option}, Type: {arg1_type}):
"{arg1_content}"

Argument 2 (Supporting option: {arg2_option}, Type: {arg2_type}):
"{arg2_content}"

Determine the relationship from Argument 1 to Argument 2:

**ATTACK**: Argument 1 contradicts, refutes, or undermines Argument 2
- Examples: 
  * "Statement is hearsay" ATTACKS "Statement is not hearsay"
  * "No exception applies" ATTACKS "Present sense impression exception applies"
  * Direct logical contradiction
  * One argument refutes the premise/conclusion of the other

**SUPPORT**: Argument 1 reinforces, strengthens, or provides evidence for Argument 2
- Examples:
  * "Statement was out of court" SUPPORTS "Statement is hearsay"
  * "Multiple witnesses confirm" SUPPORTS "Testimony is reliable"
  * One argument provides evidence for the other
  * Arguments build on same reasoning chain

**NEUTRAL**: Arguments are independent, address different aspects, or have no direct logical connection
- Examples:
  * "Statement was out of court" vs "Declarant is available" (different aspects)
  * Arguments about different legal principles
  * No semantic overlap or logical dependency

IMPORTANT RULES:
1. **Analyze SEMANTIC CONTENT**, not just the options they support
2. Two arguments supporting opposite options MAY be neutral if they address different aspects
3. Two arguments supporting the SAME option may still be neutral if independent
4. Focus on LOGICAL relationships, not just whether they're on the same "side"

Provide your analysis in JSON format:
{{
    "relation_type": "attack" | "support" | "neutral",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of the semantic relationship"
}}

Return ONLY valid JSON.
"""
        
        try:
            response = call_llm(prompt, temperature=0.2)
            
            # Parse JSON response
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            
            return ArgumentRelation(
                source_id="arg1",
                target_id="arg2",
                relation_type=result.get("relation_type", "neutral"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", "")
            )
            
        except Exception as e:
            print(f"[Warning] Failed to parse LLM response for relation analysis: {e}")
            # Fallback to heuristic if LLM fails
            return self._fallback_heuristic(arg1_option, arg1_type, arg2_option, arg2_type)
    
    def _fallback_heuristic(
        self, 
        arg1_option: str, 
        arg1_type: str, 
        arg2_option: str, 
        arg2_type: str
    ) -> ArgumentRelation:
        """
        Fallback to simple heuristic if LLM analysis fails.
        Same as original implementation.
        """
        if arg1_option != arg2_option:
            # Different options - likely attack
            return ArgumentRelation(
                source_id="arg1",
                target_id="arg2",
                relation_type="attack",
                confidence=0.7,
                reasoning="Fallback: Arguments support opposing conclusions"
            )
        elif arg1_type == arg2_type:
            # Same option, same type - likely support
            return ArgumentRelation(
                source_id="arg1",
                target_id="arg2",
                relation_type="support",
                confidence=0.6,
                reasoning="Fallback: Arguments of same type for same option"
            )
        else:
            # Same option, different type - neutral
            return ArgumentRelation(
                source_id="arg1",
                target_id="arg2",
                relation_type="neutral",
                confidence=0.5,
                reasoning="Fallback: Same option but different argument types"
            )
    
    def analyze_batch_relations(
        self,
        pairs: List[Dict],
        task_context: str = ""
    ) -> List[Dict]:
        """
        Analyze multiple argument pairs in a single LLM call (batch mode).
        
        Args:
            pairs: List of dicts with keys: pair_id, arg1_content, arg1_option, arg1_type, 
                   arg2_content, arg2_option, arg2_type
            task_context: Context about the legal task
        
        Returns:
            List of dicts with pair_id, relation_type, confidence, reasoning
        """
        if not pairs:
            return []
        
        # Build the pairs description with sanitized content
        pairs_text = ""
        for i, pair in enumerate(pairs):
            # Sanitize content: escape quotes and remove problematic characters
            arg1_content = pair['arg1_content'].replace('"', "'").replace('\n', ' ').replace('\r', ' ')[:500]
            arg2_content = pair['arg2_content'].replace('"', "'").replace('\n', ' ').replace('\r', ' ')[:500]
            
            pairs_text += f"""
--- PAIR {pair['pair_id']} ---
Argument A (Option: {pair['arg1_option']}, Type: {pair['arg1_type']}):
{arg1_content}

Argument B (Option: {pair['arg2_option']}, Type: {pair['arg2_type']}):
{arg2_content}
"""
        
        prompt = f"""You are an expert in legal argumentation and natural language inference.

Task: Analyze the semantic relationships between multiple pairs of legal arguments.

Context: {task_context if task_context else "Legal reasoning analysis"}

{pairs_text}

For EACH pair, determine the relationship from Argument A to Argument B:

**ATTACK**: Argument A contradicts, refutes, or undermines Argument B
- Direct logical contradiction
- One argument refutes the premise/conclusion of the other

**SUPPORT**: Argument A reinforces, strengthens, or provides evidence for Argument B
- One argument provides evidence for the other
- Arguments build on same reasoning chain

**NEUTRAL**: Arguments are independent, address different aspects, or have no direct logical connection
- Different legal principles with no semantic overlap
- No logical dependency between them

IMPORTANT RULES:
1. Analyze SEMANTIC CONTENT, not just the options they support
2. Two arguments supporting opposite options MAY be neutral if they address different aspects
3. Focus on LOGICAL relationships

Provide your analysis as a JSON array with one object per pair:
[
    {{
        "pair_id": "pair_0_1",
        "relation_type": "attack" | "support" | "neutral",
        "confidence": 0.0-1.0,
        "reasoning": "Brief explanation"
    }},
    ...
]

Return ONLY the valid JSON array, no other text.
"""
        
        try:
            response = call_llm(prompt, temperature=0.2, max_tokens=4096)
            
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            # Additional cleaning: try to extract JSON array if response has extra text
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            results = json.loads(response)
            
            if isinstance(results, list):
                return results
            else:
                print(f"[Warning] Batch response is not a list, falling back to individual analysis")
                return []
                
        except Exception as e:
            print(f"[Warning] Batch analysis failed: {e}")
            return []
    
    def analyze_all_relations(
        self, 
        arguments: List,
        task_context: str = "",
        use_semantic: bool = True,
        batch_size: int = 10
    ) -> Tuple[Dict[Tuple[str, str], ArgumentRelation], bool]:
        """
        Analyze relations between all pairs of arguments.
        
        Args:
            arguments: List of Argument objects from state
            task_context: Context about the legal task
            use_semantic: If True, use LLM-based semantic analysis; if False, use heuristics only
            batch_size: Number of pairs to analyze in each LLM call (default: 10)
        
        Returns:
            Tuple of:
                - Dictionary mapping (arg_i_id, arg_j_id) to ArgumentRelation
                - Boolean indicating if heuristic fallback was used (True = heuristic used)
        """
        relations = {}
        used_heuristic_fallback = False
        # Only analyze each pair once (i < j), not both directions
        # This cuts API calls in half: n*(n-1)/2 instead of n*(n-1)
        total_pairs = len(arguments) * (len(arguments) - 1) // 2
        
        print(f"[Semantic Analysis] Analyzing {total_pairs} argument pairs...")
        print(f"[Semantic Analysis] Mode: {'LLM-based semantic (BATCH)' if use_semantic else 'Heuristic only'}")
        
        if use_semantic and total_pairs > 0:
            # BATCH MODE: Collect all pairs first, then process in batches
            all_pairs = []
            pair_info = []  # Store (i, j) indices for mapping results back
            
            for i, arg_i in enumerate(arguments):
                for j, arg_j in enumerate(arguments):
                    if i >= j:
                        continue
                    
                    pair_id = f"pair_{i}_{j}"
                    all_pairs.append({
                        'pair_id': pair_id,
                        'arg1_content': arg_i.content,
                        'arg1_option': arg_i.parent_option,
                        'arg1_type': arg_i.argument_type,
                        'arg2_content': arg_j.content,
                        'arg2_option': arg_j.parent_option,
                        'arg2_type': arg_j.argument_type
                    })
                    pair_info.append((i, j, arg_i, arg_j))
            
            # Process in batches
            num_batches = (len(all_pairs) + batch_size - 1) // batch_size
            print(f"[Semantic Analysis] Processing {num_batches} batches (batch_size={batch_size})")
            
            processed = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_pairs))
                batch_pairs = all_pairs[start_idx:end_idx]
                batch_pair_info = pair_info[start_idx:end_idx]
                
                print(f"[Semantic Analysis] Batch {batch_idx + 1}/{num_batches}: pairs {start_idx}-{end_idx-1}")
                
                # Call batch analysis
                batch_results = self.analyze_batch_relations(batch_pairs, task_context)
                
                # Map results back to pairs
                result_map = {r['pair_id']: r for r in batch_results}
                
                for idx, (i, j, arg_i, arg_j) in enumerate(batch_pair_info):
                    pair_id = f"pair_{i}_{j}"
                    arg_i_id = f"arg_{i}"
                    arg_j_id = f"arg_{j}"
                    
                    if pair_id in result_map:
                        result = result_map[pair_id]
                        relation = ArgumentRelation(
                            source_id=arg_i_id,
                            target_id=arg_j_id,
                            relation_type=result.get("relation_type", "neutral"),
                            confidence=float(result.get("confidence", 0.5)),
                            reasoning=result.get("reasoning", "")
                        )
                    else:
                        # Fallback for missing results
                        used_heuristic_fallback = True
                        print(f"[Warning] No result for {pair_id}, using heuristic fallback")
                        relation = self._fallback_heuristic(
                            arg_i.parent_option,
                            arg_i.argument_type,
                            arg_j.parent_option,
                            arg_j.argument_type
                        )
                        relation.source_id = arg_i_id
                        relation.target_id = arg_j_id
                    
                    relations[(arg_i_id, arg_j_id)] = relation
                    processed += 1
                
                print(f"[Semantic Analysis] Progress: {processed}/{total_pairs} pairs analyzed")
        
        else:
            # HEURISTIC MODE: Process one by one (fast, no LLM calls)
            processed = 0
            for i, arg_i in enumerate(arguments):
                for j, arg_j in enumerate(arguments):
                    if i >= j:
                        continue
                    
                    arg_i_id = f"arg_{i}"
                    arg_j_id = f"arg_{j}"
                    
                    relation = self._fallback_heuristic(
                        arg_i.parent_option,
                        arg_i.argument_type,
                        arg_j.parent_option,
                        arg_j.argument_type
                    )
                    relation.source_id = arg_i_id
                    relation.target_id = arg_j_id
                    
                    relations[(arg_i_id, arg_j_id)] = relation
                    processed += 1
            
            if processed > 0:
                print(f"[Semantic Analysis] Progress: {processed}/{total_pairs} pairs analyzed")
        
        # Print summary
        attack_count = sum(1 for r in relations.values() if r.relation_type == "attack")
        support_count = sum(1 for r in relations.values() if r.relation_type == "support")
        neutral_count = sum(1 for r in relations.values() if r.relation_type == "neutral")
        
        print(f"\n[Semantic Analysis] Results:")
        if total_pairs > 0:
            print(f"  Attack relations:  {attack_count} ({attack_count/total_pairs*100:.1f}%)")
            print(f"  Support relations: {support_count} ({support_count/total_pairs*100:.1f}%)")
            print(f"  Neutral relations: {neutral_count} ({neutral_count/total_pairs*100:.1f}%)")
        else:
            print(f"  No argument pairs to analyze (0 or 1 arguments)")
        
        if used_heuristic_fallback:
            print(f"  ⚠️ HEURISTIC FALLBACK WAS USED - some pairs could not be analyzed by LLM")
        
        return relations, used_heuristic_fallback
    
    def filter_relations_by_threshold(
        self, 
        relations: Dict[Tuple[str, str], ArgumentRelation]
    ) -> Dict[Tuple[str, str], ArgumentRelation]:
        """
        Filter relations based on confidence thresholds.
        Low-confidence relations are converted to neutral.
        """
        filtered = {}
        
        for key, relation in relations.items():
            if relation.relation_type == "attack" and relation.confidence < self.threshold_attack:
                # Convert low-confidence attack to neutral
                relation.relation_type = "neutral"
            elif relation.relation_type == "support" and relation.confidence < self.threshold_support:
                # Convert low-confidence support to neutral
                relation.relation_type = "neutral"
            
            filtered[key] = relation
        
        return filtered


def analyze_argument_relations_semantic(
    arguments: List,
    task_context: str = "",
    use_semantic: bool = True,
    threshold_attack: float = 0.6,
    threshold_support: float = 0.6
) -> Tuple[Dict[Tuple[str, str], ArgumentRelation], bool]:
    """
    Convenience function to analyze argument relations with semantic analysis.
    
    Args:
        arguments: List of Argument objects
        task_context: Description of the legal task for context
        use_semantic: Use LLM-based semantic analysis (True) or heuristics (False)
        threshold_attack: Confidence threshold for attack relations
        threshold_support: Confidence threshold for support relations
    
    Returns:
        Tuple of:
            - Dictionary of argument relations
            - Boolean indicating if heuristic fallback was used (True = some pairs used heuristic)
    """
    analyzer = SemanticRelationAnalyzer(
        threshold_attack=threshold_attack,
        threshold_support=threshold_support
    )
    
    relations, used_heuristic = analyzer.analyze_all_relations(
        arguments=arguments,
        task_context=task_context,
        use_semantic=use_semantic
    )
    
    # Filter by confidence thresholds
    relations = analyzer.filter_relations_by_threshold(relations)
    
    return relations, used_heuristic
