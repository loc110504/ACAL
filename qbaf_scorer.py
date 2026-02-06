import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import math
import json
import re
from semantic_relation_analyzer import analyze_argument_relations_semantic
from llm_caller import call_llm


@dataclass
class QBAFArgument:
    """Argument node in QBAF graph"""
    id: str
    content: str
    base_score: float  # Initial strength from LLM (0-1)
    parent_option: str  # "Yes" or "No"
    argument_type: str  # "support" or "attack"
    agent_role: Optional[str] = None
    agent_name: Optional[str] = None
    is_decision_node: bool = False  # True for Yes/No decision nodes
    
    # QBAF computed scores
    final_score: float = 0.0
    support_impact: float = 0.0
    attack_impact: float = 0.0
    
    # Relations with other arguments
    supports: List[str] = field(default_factory=list)  # IDs of arguments this supports
    attacks: List[str] = field(default_factory=list)   # IDs of arguments this attacks
    supported_by: List[str] = field(default_factory=list)
    attacked_by: List[str] = field(default_factory=list)


class QBAFScorer:
    """
    Implements QBAF with various gradual semantics for computing argument strength.
    
    Semantics supported:
    - weighted_sum: Σ(support) - Σ(attack)
    - weighted_product: base * Π(1 + support) / Π(1 + attack)
    - euler_based: Uses Euler-based aggregation function
    - df_quad: DF-QuAD semantics (Discontinuity-Free Quantitative Argumentation Debate)
    - quadratic_energy: QE semantics - minimizes quadratic energy function (Potyka, 2018)
    """
    
    def __init__(self, semantics: str = "quadratic_energy", convergence_threshold: float = 0.001, max_iterations: int = 100):
        """
        Args:
            semantics: Type of gradual semantics to use
            convergence_threshold: Stop iteration when change < threshold
            max_iterations: Maximum number of iterations
        """
        self.semantics = semantics
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.arguments: Dict[str, QBAFArgument] = {}
        
    def add_argument(self, arg: QBAFArgument) -> None:
        """Add an argument to the framework"""
        self.arguments[arg.id] = arg
        
    def add_support(self, supporter_id: str, supported_id: str) -> None:
        """Add a support relation between two arguments"""
        if supporter_id in self.arguments and supported_id in self.arguments:
            self.arguments[supporter_id].supports.append(supported_id)
            self.arguments[supported_id].supported_by.append(supporter_id)
            
    def add_attack(self, attacker_id: str, attacked_id: str) -> None:
        """Add an attack relation between two arguments"""
        if attacker_id in self.arguments and attacked_id in self.arguments:
            self.arguments[attacker_id].attacks.append(attacked_id)
            self.arguments[attacked_id].attacked_by.append(attacker_id)
    
    def build_argument_graph_from_state(
        self, 
        arguments: List, 
        option_mapping: Dict[str, str] = None,  # Legacy param, no longer needed
        use_semantic_analysis: bool = False,
        task_context: str = "",
        claim: str = "The claim is true"
    ) -> bool:
        """
        Build QBAF graph following the standard model (Rago et al. 2016).
        
        Standard QBAF Structure:
        - Central CLAIM node (base score 0.5)
        - Evidence/Argument nodes that SUPPORT or ATTACK the claim
        - Inter-argument relations (support/attack between arguments)
        - Final claim score >= threshold → Yes, else No
        
        Two modes for inter-argument relations:
        1. Heuristic (use_semantic_analysis=False): 
           - Support arguments support each other
           - Attack arguments support each other  
           - Support and attack arguments attack each other
        
        2. Semantic (use_semantic_analysis=True): LLM-based NLI analysis
           - Analyzes actual content to determine attack/support/neutral
           - More accurate but slower and uses more API calls
        
        Args:
            arguments: List of Argument objects from your state
            option_mapping: DEPRECATED - no longer used in standard QBAF
            use_semantic_analysis: Use LLM-based semantic analysis (default: False)
            task_context: Description of legal task for semantic analysis
            claim: The central thesis being evaluated
        """
        # Clear existing
        self.arguments = {}
        
        # Step 1: Create central CLAIM node
        # This is the thesis being evaluated (e.g., "This is hearsay")
        claim_node = QBAFArgument(
            id="claim",
            content=claim,
            base_score=0.5,  # Neutral starting point
            parent_option=None,
            argument_type="claim",
            is_decision_node=True
        )
        self.add_argument(claim_node)
        
        # Step 2: Create argument nodes (Evidence)
        for i, arg in enumerate(arguments):
            qbaf_arg = QBAFArgument(
                id=f"arg_{i}",
                content=arg.content,
                base_score=arg.validity_score if arg.validity_score is not None else 0.5,
                parent_option=None,  # No longer used
                argument_type=arg.argument_type,  # "support" or "attack"
                agent_role=arg.agent_role,
                agent_name=arg.agent_name,
                is_decision_node=False
            )
            self.add_argument(qbaf_arg)
        
        # Step 3: Connect arguments to the CLAIM node
        # SIMPLE STANDARD MODEL:
        #   - argument_type="support" → SUPPORTS the claim (green arrow to claim)
        #   - argument_type="attack" → ATTACKS the claim (red arrow to claim)
        for i, arg in enumerate(arguments):
            arg_id = f"arg_{i}"
            claim_id = "claim"
            
            if arg.argument_type == "support":
                self.add_support(arg_id, claim_id)
            elif arg.argument_type == "attack":
                self.add_attack(arg_id, claim_id)
        
        # Step 4: Infer attack/support relations between arguments
        used_heuristic_fallback = False
        if use_semantic_analysis:
            print(f"[QBAF] Using SEMANTIC ANALYSIS for inter-argument relations...")
            used_heuristic_fallback = self._build_relations_semantic(arguments, task_context)
        else:
            print(f"[QBAF] Using HEURISTIC RULES for inter-argument relations...")
            self._build_relations_heuristic()
            used_heuristic_fallback = True  # Full heuristic mode
        
        print(f"[QBAF] Built graph with {len(self.arguments)} nodes (1 claim + {len(arguments)} arguments)")
        self._print_graph_statistics()
        
        # Return whether heuristic fallback was used
        return used_heuristic_fallback
    
    def _build_relations_heuristic(self) -> None:
        """
        Build inter-argument relations using simple heuristic rules.
        
        Standard QBAF heuristics (BIDIRECTIONAL):
        - Support arguments support each other (mutual reinforcement)
        - Attack arguments support each other (mutual reinforcement of opposition)
        - Support and Attack arguments attack each other (mutual conflict)
        
        All relations are bidirectional to ensure symmetric treatment.
        """
        # Only consider regular arguments, not the claim node
        arg_list = [arg for arg in self.arguments.values() if not arg.is_decision_node]
        
        for i, arg_i in enumerate(arg_list):
            for j, arg_j in enumerate(arg_list):
                if i >= j:  # Avoid duplicate processing and self-relations
                    continue
                
                # Same type → MUTUAL support (both directions)
                # Different type → MUTUAL attack (both directions)
                if arg_i.argument_type == arg_j.argument_type:
                    # Both support OR both attack → they reinforce each other (bidirectional)
                    self.add_support(arg_i.id, arg_j.id)  # i supports j
                    self.add_support(arg_j.id, arg_i.id)  # j supports i
                else:
                    # One support, one attack → they conflict (bidirectional)
                    self.add_attack(arg_i.id, arg_j.id)  # i attacks j
                    self.add_attack(arg_j.id, arg_i.id)  # j attacks i
    
    def _build_relations_semantic(self, arguments: List, task_context: str) -> bool:
        """
        Build relations using LLM-based semantic analysis.
        SLOW but more accurate - analyzes actual argument content.
        
        Relations are made BIDIRECTIONAL:
        - If A attacks B, we also add B attacks A
        - If A supports B, we also add B supports A
        This ensures symmetric treatment in QBAF calculations.
        
        Returns:
            Boolean indicating if heuristic fallback was used for any pairs
        """
        # Analyze all relations using semantic analyzer
        relations, used_heuristic = analyze_argument_relations_semantic(
            arguments=arguments,
            task_context=task_context,
            use_semantic=True,
            threshold_attack=0.6,
            threshold_support=0.6
        )
        
        # Add relations to QBAF graph (BIDIRECTIONAL)
        for (src_id, tgt_id), relation in relations.items():
            if relation.relation_type == "attack":
                self.add_attack(src_id, tgt_id)  # src attacks tgt
                self.add_attack(tgt_id, src_id)  # tgt attacks src (bidirectional)
            elif relation.relation_type == "support":
                self.add_support(src_id, tgt_id)  # src supports tgt
                self.add_support(tgt_id, src_id)  # tgt supports src (bidirectional)
            # neutral relations are not added
        
        return used_heuristic
    
    def resolve_argument_clashes(
        self, 
        arguments: List,
        claim: str,
        case_text: str,
        task_context: str = "",
        provider: str = "",
        clash_trigger_threshold: float = 0.2,
        base_adjustment: float = 0.15
    ) -> Dict[str, float]:
        """
        Resolve clashes between opposing arguments using LLM analysis.
        
        ONLY processes pairs that have ATTACK RELATIONS in the QBAF graph.
        This respects semantic analysis results - if two args don't conflict
        in the graph, they won't be compared.
        
        ORDER-INDEPENDENT BATCH EVALUATION:
        1. Find pairs with attack relations in QBAF graph
        2. Evaluate pairs using ORIGINAL scores (if similar)
        3. Count wins/losses for each argument across all its clashes
        4. Calculate adjustment based on win RATE (not raw count)
        
        Args:
            arguments: Original list of Argument objects
            claim: The central claim being evaluated
            case_text: The case/scenario text for context
            task_context: Task description (e.g., "hearsay analysis")
            provider: LLM provider to use ("gpt", "gemini", etc.)
            clash_trigger_threshold: Only resolve if |score_diff| < threshold (default 0.2)
            base_adjustment: Maximum adjustment for worst performer (default 0.15)
        
        Returns:
            Dictionary mapping argument IDs to their adjusted scores
        """
        print(f"\n[CLASH] Batch clash resolution (threshold={clash_trigger_threshold})...")
        
        # Build lookup from argument ID to original Argument object and score
        arg_lookup = {}
        for i, arg in enumerate(arguments):
            arg_id = f"arg_{i}"
            if arg_id in self.arguments:
                arg_lookup[arg_id] = {
                    "arg": arg,
                    "qbaf": self.arguments[arg_id],
                    "original_score": self.arguments[arg_id].base_score
                }
        
        # Find pairs that have ATTACK relations in the QBAF graph
        # Only these pairs should be evaluated for clash resolution
        attack_pairs = []
        for arg_id, qbaf_arg in self.arguments.items():
            if qbaf_arg.is_decision_node:
                continue
            for attacked_id in qbaf_arg.attacks:
                # Skip attacks on the claim node
                if attacked_id == "claim":
                    continue
                # Only consider cross-type attacks (support vs attack)
                if arg_id in arg_lookup and attacked_id in arg_lookup:
                    arg1_type = arg_lookup[arg_id]["arg"].argument_type
                    arg2_type = arg_lookup[attacked_id]["arg"].argument_type
                    if arg1_type != arg2_type:
                        # Add pair (sorted to avoid duplicates)
                        pair = tuple(sorted([arg_id, attacked_id]))
                        if pair not in attack_pairs:
                            attack_pairs.append(pair)
        
        if not attack_pairs:
            print("[CLASH] No attack relations found between arguments")
            return {}
        
        print(f"[CLASH] Found {len(attack_pairs)} attack relation pairs to evaluate")
        
        # Track wins and losses for each argument (order-independent)
        arg_stats = {arg_id: {"wins": 0, "losses": 0, "total_clashes": 0} 
                     for arg_id in self.arguments.keys()}
        
        clash_results = []
        skipped_count = 0
        resolved_count = 0
        no_relation_count = 0
        
        # PHASE 1: Evaluate pairs WITH ATTACK RELATIONS using ORIGINAL scores
        print(f"[CLASH] Phase 1: Evaluating pairs with attack relations...")
        for arg1_id, arg2_id in attack_pairs:
            # Determine which is support and which is attack
            arg1_data = arg_lookup[arg1_id]
            arg2_data = arg_lookup[arg2_id]
            
            if arg1_data["arg"].argument_type == "support":
                sup_id, sup_data = arg1_id, arg1_data
                att_id, att_data = arg2_id, arg2_data
            else:
                sup_id, sup_data = arg2_id, arg2_data
                att_id, att_data = arg1_id, arg1_data
            
            sup_original_score = sup_data["original_score"]
            att_original_score = att_data["original_score"]
            
            # Check if this pair needs resolution (using ORIGINAL scores)
            score_diff = abs(sup_original_score - att_original_score)
            
            if score_diff >= clash_trigger_threshold:
                # Clear winner - no clash needed
                skipped_count += 1
                winner = "support" if sup_original_score > att_original_score else "attack"
                print(f"  [{sup_id} vs {att_id}] SKIP - clear winner ({winner}, diff={score_diff:.2f})")
                continue
            
            # Scores are similar - call LLM to resolve
            resolved_count += 1
            print(f"  [{sup_id} vs {att_id}] RESOLVING ({sup_original_score:.2f} vs {att_original_score:.2f})")
            
            winner_id, _ = self._resolve_single_clash(
                sup_id=sup_id,
                sup_content=sup_data["arg"].content,
                sup_score=sup_original_score,  # Always use ORIGINAL score
                att_id=att_id,
                att_content=att_data["arg"].content,
                att_score=att_original_score,  # Always use ORIGINAL score
                claim=claim,
                case_text=case_text,
                task_context=task_context,
                provider=provider
            )
            
            # Record win/loss (don't adjust scores yet!)
            loser_id = att_id if winner_id == sup_id else sup_id
            arg_stats[winner_id]["wins"] += 1
            arg_stats[winner_id]["total_clashes"] += 1
            arg_stats[loser_id]["losses"] += 1
            arg_stats[loser_id]["total_clashes"] += 1
            
            clash_results.append({
                "support": sup_id,
                "attack": att_id,
                "winner": winner_id,
                "loser": loser_id
            })
        
        print(f"\n[CLASH] Phase 1 complete: {resolved_count} resolved, {skipped_count} skipped")
        
        # PHASE 2: Calculate adjustments based on WIN RATE
        # Now symmetric: winners get bonus, losers get penalty
        print(f"[CLASH] Phase 2: Calculating adjustments based on win rate...")
        score_adjustments = {}
        
        for arg_id, stats in arg_stats.items():
            if stats["total_clashes"] == 0:
                continue  # No clashes for this argument
            
            win_rate = stats["wins"] / stats["total_clashes"]
            # win_rate = 1.0 → full bonus (always won)
            # win_rate = 0.5 → no change (won half, lost half)
            # win_rate = 0.0 → full penalty (always lost)
            
            # SYMMETRIC adjustment formula:
            # - win_rate > 0.5 → positive adjustment (bonus)
            # - win_rate < 0.5 → negative adjustment (penalty)
            # - win_rate = 0.5 → no adjustment
            adjustment = base_adjustment * (win_rate - 0.5) * 2  # Scale to [-base, +base]
            
            if abs(adjustment) > 0.001:  # Only record meaningful adjustments
                score_adjustments[arg_id] = adjustment
                direction = "+" if adjustment > 0 else ""
                print(f"  {arg_id}: {stats['wins']}W/{stats['losses']}L (rate={win_rate:.2f}) → adj={direction}{adjustment:.3f}")
        
        # PHASE 3: Apply adjustments to base scores
        print(f"\n[CLASH] Phase 3: Applying adjustments...")
        for arg_id, adjustment in score_adjustments.items():
            if arg_id in self.arguments and not self.arguments[arg_id].is_decision_node:
                old_score = self.arguments[arg_id].base_score
                # Clamp to [0.1, 0.95] to prevent extreme scores
                new_score = max(0.1, min(0.95, old_score + adjustment))
                self.arguments[arg_id].base_score = new_score
                print(f"  {arg_id}: {old_score:.3f} → {new_score:.3f}")
        
        # Print summary
        support_wins = sum(1 for r in clash_results 
                          if r["winner"] in arg_lookup and 
                          arg_lookup[r["winner"]]["arg"].argument_type == "support")
        attack_wins = len(clash_results) - support_wins
        print(f"\n[CLASH] Final: {support_wins} support wins, {attack_wins} attack wins")
        
        return score_adjustments
    
    def _resolve_single_clash(
        self,
        sup_id: str,
        sup_content: str,
        sup_score: float,
        att_id: str,
        att_content: str,
        att_score: float,
        claim: str,
        case_text: str,
        task_context: str,
        provider: str
    ) -> Tuple[str, float]:
        """
        Resolve a single clash between a support and attack argument.
        
        Returns:
            Tuple of (winner_id, adjustment_amount)
        """
        prompt = f"""You are a legal reasoning expert. Analyze these two conflicting arguments about a legal claim.

CLAIM: {claim}

CASE DETAILS:
{case_text[:1500]}

TASK CONTEXT: {task_context}

ARGUMENT A (SUPPORTS the claim):
"{sup_content}"
Current Score: {sup_score:.2f}

ARGUMENT B (ATTACKS the claim):
"{att_content}"
Current Score: {att_score:.2f}

These arguments directly conflict. Analyze which argument is STRONGER and more legally sound in the context of this specific case.

Consider:
1. Which argument is more directly relevant to the case facts?
2. Which argument applies the correct legal standard?
3. Which argument has stronger logical reasoning?
4. Does either argument misinterpret the facts or law?

Respond in JSON format:
{{
    "winner": "A" or "B",
    "winner_reason": "Brief explanation of why this argument is stronger",
    "loser_weakness": "Brief explanation of the weaker argument's flaw",
    "adjustment": 0.1 to 0.4 (how much to reduce the loser's score - use higher values for clear defeats)
}}

Only output the JSON, no other text."""

        try:
            response = call_llm(prompt, temperature=0.2, max_tokens=500)
            
            # Parse JSON response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                winner = result.get("winner", "A")
                adjustment = float(result.get("adjustment", 0.2))
                adjustment = max(0.1, min(0.4, adjustment))  # Clamp to [0.1, 0.4]
                
                winner_id = sup_id if winner == "A" else att_id
                print(f"  [{sup_id} vs {att_id}] Winner: {winner} (adj: {adjustment:.2f}) - {result.get('winner_reason', '')[:60]}...")
                return winner_id, adjustment
            else:
                print(f"  [{sup_id} vs {att_id}] Failed to parse response, defaulting to tie")
                return sup_id, 0.1  # Default: small adjustment
                
        except Exception as e:
            print(f"  [{sup_id} vs {att_id}] Error: {e}, defaulting to tie")
            return sup_id, 0.1

    def compute_final_scores(self) -> Dict[str, float]:
        """
        Compute final scores for all arguments using selected semantics.
        Returns dictionary mapping argument IDs to final scores.
        """
        if self.semantics == "weighted_sum":
            return self._compute_weighted_sum()
        elif self.semantics == "weighted_product":
            return self._compute_weighted_product()
        elif self.semantics == "euler_based":
            return self._compute_euler_based()
        elif self.semantics == "df_quad":
            return self._compute_df_quad()
        elif self.semantics == "quadratic_energy":
            return self._compute_quadratic_energy()
        else:
            raise ValueError(f"Unknown semantics: {self.semantics}")
    
    def _compute_weighted_sum(self) -> Dict[str, float]:
        """
        Simple weighted sum semantics:
        final_score = base_score + Σ(support_scores) - Σ(attack_scores)
        """
        scores = {arg_id: arg.base_score for arg_id, arg in self.arguments.items()}
        
        iteration = 0
        while iteration < self.max_iterations:
            new_scores = {}
            max_change = 0
            
            for arg_id, arg in self.arguments.items():
                support_sum = sum(scores[sup_id] * scores[arg_id] for sup_id in arg.supported_by)
                attack_sum = sum(scores[att_id] * scores[arg_id] for att_id in arg.attacked_by)
                
                new_score = arg.base_score + 0.3 * support_sum - 0.3 * attack_sum
                new_score = max(0, min(1, new_score))  # Clamp to [0, 1]
                
                new_scores[arg_id] = new_score
                max_change = max(max_change, abs(new_score - scores[arg_id]))
            
            scores = new_scores
            iteration += 1
            
            if max_change < self.convergence_threshold:
                print(f"[QBAF] Converged after {iteration} iterations")
                break
        
        # Update argument objects
        for arg_id, score in scores.items():
            self.arguments[arg_id].final_score = score
        
        return scores
    
    def _compute_weighted_product(self) -> Dict[str, float]:
        """
        Product-based semantics:
        final_score = base_score * Π(1 + α*support) / Π(1 + β*attack)
        """
        alpha = 0.3  # Support weight
        beta = 0.3   # Attack weight
        
        scores = {arg_id: arg.base_score for arg_id, arg in self.arguments.items()}
        
        iteration = 0
        while iteration < self.max_iterations:
            new_scores = {}
            max_change = 0
            
            for arg_id, arg in self.arguments.items():
                support_product = np.prod([1 + alpha * scores[sup_id] for sup_id in arg.supported_by]) if arg.supported_by else 1.0
                attack_product = np.prod([1 + beta * scores[att_id] for att_id in arg.attacked_by]) if arg.attacked_by else 1.0
                
                new_score = arg.base_score * (support_product / attack_product)
                new_score = max(0, min(1, new_score))
                
                new_scores[arg_id] = new_score
                max_change = max(max_change, abs(new_score - scores[arg_id]))
            
            scores = new_scores
            iteration += 1
            
            if max_change < self.convergence_threshold:
                print(f"[QBAF] Converged after {iteration} iterations")
                break
        
        for arg_id, score in scores.items():
            self.arguments[arg_id].final_score = score
        
        return scores
    
    def _compute_euler_based(self) -> Dict[str, float]:
        """
        Euler-based gradual semantics (Baroni et al., 2015):
        Uses exponential aggregation for smooth transitions
        """
        gamma_support = 0.5  # Support influence
        gamma_attack = 0.5   # Attack influence
        
        scores = {arg_id: arg.base_score for arg_id, arg in self.arguments.items()}
        
        iteration = 0
        while iteration < self.max_iterations:
            new_scores = {}
            max_change = 0
            
            for arg_id, arg in self.arguments.items():
                support_sum = sum(scores[sup_id] for sup_id in arg.supported_by)
                attack_sum = sum(scores[att_id] for att_id in arg.attacked_by)
                
                # Euler-based formula
                support_factor = 1 / (1 + math.exp(-gamma_support * support_sum))
                attack_factor = 1 / (1 + math.exp(gamma_attack * attack_sum))
                
                new_score = arg.base_score * support_factor * attack_factor
                new_score = max(0, min(1, new_score))
                
                new_scores[arg_id] = new_score
                max_change = max(max_change, abs(new_score - scores[arg_id]))
            
            scores = new_scores
            iteration += 1
            
            if max_change < self.convergence_threshold:
                print(f"[QBAF] Converged after {iteration} iterations")
                break
        
        for arg_id, score in scores.items():
            self.arguments[arg_id].final_score = score
        
        return scores
    
    @staticmethod
    def _dfquad_F(values: List[float]) -> float:
        """
        DF-QuAD aggregation F
          if n=0 -> 0
          else   -> 1 - Π_i (1 - v_i)
        """
        if not values:
            return 0.0
        prod = 1.0
        for v in values:
            v = max(0.0, min(1.0, float(v)))
            prod *= (1.0 - v)
        return 1.0 - prod
    
    @staticmethod
    def _dfquad_C(v0: float, va: float, vs: float, tol: float = 1e-12) -> float:
        """
        DF-QuAD combination C
          if va == vs: C = v0
          elif va > vs: C = v0 - (v0 * |vs - va|)
          else:         C = v0 + ((1 - v0) * |vs - va|)
        """
        v0 = max(0.0, min(1.0, float(v0)))
        va = max(0.0, min(1.0, float(va)))
        vs = max(0.0, min(1.0, float(vs)))

        diff = abs(vs - va)
        if abs(va - vs) <= tol:
            out = v0
        elif va > vs:
            out = v0 - (v0 * diff)
        else:
            out = v0 + ((1.0 - v0) * diff)

        return max(0.0, min(1.0, out))
    
    def _compute_df_quad(self) -> Dict[str, float]:
        """
        DF-QuAD (Discontinuity-Free Quantitative Argumentation Debate) semantics.
        Proper implementation based on Rago et al. "Discontinuity-Free Decision Support 
        with Quantitative Argumentation Debates" (KR 2016)
        
        σ(α) = C(v0, F(attackers), F(supporters))
        where:
          F(v1..vn) = 0 if n=0 else 1 - Π_i (1 - vi)
          C is piecewise combination function
        """
        scores = {arg_id: arg.base_score for arg_id, arg in self.arguments.items()}
        
        iteration = 0
        while iteration < self.max_iterations:
            new_scores = {}
            max_change = 0
            
            for arg_id, arg in self.arguments.items():
                v0 = float(arg.base_score)
                
                # Get attacker and supporter strength values
                att_vals = [scores[att_id] for att_id in arg.attacked_by]
                sup_vals = [scores[sup_id] for sup_id in arg.supported_by]
                
                # Apply DF-QuAD aggregation F
                va = self._dfquad_F(att_vals)
                vs = self._dfquad_F(sup_vals)
                
                # Apply DF-QuAD combination C
                new_score = self._dfquad_C(v0, va, vs)
                
                new_scores[arg_id] = new_score
                max_change = max(max_change, abs(new_score - scores[arg_id]))
            
            scores = new_scores
            iteration += 1
            
            if max_change < self.convergence_threshold:
                print(f"[QBAF] DF-QuAD converged after {iteration} iterations")
                break
        
        # Store impacts
        for arg_id, arg in self.arguments.items():
            if arg.supported_by:
                arg.support_impact = sum(scores[sup_id] for sup_id in arg.supported_by) / len(arg.supported_by)
            if arg.attacked_by:
                arg.attack_impact = sum(scores[att_id] for att_id in arg.attacked_by) / len(arg.attacked_by)
            arg.final_score = scores[arg_id]
        
        return scores
    
    def _compute_quadratic_energy(self) -> Dict[str, float]:
        """
        Quadratic Energy (QE) semantics based on Potyka (2018).
        
        Minimizes a quadratic energy function that balances:
        1. Staying close to the base score (intrinsic strength)
        2. Being influenced by supporters (increase strength)
        3. Being influenced by attackers (decrease strength)
        
        The energy function is:
        E(σ) = Σ_a [ (σ(a) - τ(a))² + Σ_{b attacks a} σ(a) * σ(b) 
                                     - Σ_{c supports a} σ(a) * σ(c) ]
        
        Taking the gradient with respect to σ(a):
        ∂E/∂σ(a) = 2(σ(a) - τ(a)) + Σ_{b attacks a} σ(b) - Σ_{c supports a} σ(c)
        
        The update rule uses gradient descent with learning rate α:
        σ'(a) = σ(a) - α * ∂E/∂σ(a)
        
        Simplified (rearranged):
        σ'(a) = (1 - 2α) * σ(a) + 2α * τ(a) + α * (Σ supporters - Σ attackers)
        
        With α = 0.5, this simplifies to:
        σ'(a) = τ(a) + 0.5 * (Σ_{c supports a} σ(c) - Σ_{b attacks a} σ(b))
        
        We use a smaller learning rate to prevent oscillation and extreme values.
        The result is clamped to [0, 1] after each iteration.
        """
        # Learning rate for gradient descent (smaller = more stable, less extreme)
        alpha = 0.1
        
        # Initialize with base scores
        scores = {arg_id: arg.base_score for arg_id, arg in self.arguments.items()}
        
        iteration = 0
        while iteration < self.max_iterations:
            new_scores = {}
            max_change = 0
            
            for arg_id, arg in self.arguments.items():
                base = float(arg.base_score)
                current = scores[arg_id]
                
                # Calculate support influence: sum of supporter scores
                support_sum = sum(scores[sup_id] for sup_id in arg.supported_by)
                
                # Calculate attack influence: sum of attacker scores
                attack_sum = sum(scores[att_id] for att_id in arg.attacked_by)
                
                # Normalize by count to prevent extreme values with many relations
                num_supporters = len(arg.supported_by)
                num_attackers = len(arg.attacked_by)
                
                # Average influence (normalized)
                support_avg = support_sum / num_supporters if num_supporters > 0 else 0
                attack_avg = attack_sum / num_attackers if num_attackers > 0 else 0
                
                # Gradient: ∂E/∂σ(a) = 2(σ(a) - τ(a)) + attacks - supports
                gradient = 2 * (current - base) + attack_avg - support_avg
                
                # Gradient descent update: σ' = σ - α * gradient
                new_score = current - alpha * gradient
                
                # Clamp to [0, 1]
                new_score = max(0.0, min(1.0, new_score))
                
                new_scores[arg_id] = new_score
                max_change = max(max_change, abs(new_score - scores[arg_id]))
            
            scores = new_scores
            iteration += 1
            
            if max_change < self.convergence_threshold:
                print(f"[QBAF] Quadratic Energy converged after {iteration} iterations")
                break
        
        if iteration >= self.max_iterations:
            print(f"[QBAF] Quadratic Energy reached max iterations ({self.max_iterations})")
        
        # Store impacts and final scores
        for arg_id, arg in self.arguments.items():
            if arg.supported_by:
                arg.support_impact = sum(scores[sup_id] for sup_id in arg.supported_by) / len(arg.supported_by)
            if arg.attacked_by:
                arg.attack_impact = sum(scores[att_id] for att_id in arg.attacked_by) / len(arg.attacked_by)
            arg.final_score = scores[arg_id]
        
        return scores

    def get_option_scores(self) -> Dict[str, float]:
        """
        Get the decision node score and compute argument statistics.
        
        The single decision node's final_score represents the claim strength.
        Score >= threshold → Yes; Score < threshold → No
        
        Returns:
            Dictionary with claim score and argument statistics
        """
        result = {}
        
        # Get claim node score
        claim_node = None
        for arg in self.arguments.values():
            if arg.is_decision_node:
                claim_node = arg
                break
        
        if claim_node:
            result["claim"] = {
                "claim_score": claim_node.final_score,
                "base_score": claim_node.base_score,
                "support_impact": claim_node.support_impact,
                "attack_impact": claim_node.attack_impact,
                "content": claim_node.content
            }
        
        # Compute argument statistics by type (support vs attack)
        support_args = []
        attack_args = []
        
        for arg in self.arguments.values():
            if not arg.is_decision_node:
                if arg.argument_type == "support":
                    support_args.append(arg)
                elif arg.argument_type == "attack":
                    attack_args.append(arg)
        
        result["support_arguments"] = {
            "count": len(support_args),
            "total_score": sum(a.final_score for a in support_args),
            "average_score": sum(a.final_score for a in support_args) / len(support_args) if support_args else 0.0
        }
        
        result["attack_arguments"] = {
            "count": len(attack_args),
            "total_score": sum(a.final_score for a in attack_args),
            "average_score": sum(a.final_score for a in attack_args) / len(attack_args) if attack_args else 0.0
        }
        
        return result
    
    def get_top_arguments(self, arg_type: str = None, top_k: int = 5) -> List[QBAFArgument]:
        """Get top-k arguments by final score, optionally filtered by argument type"""
        filtered = [arg for arg in self.arguments.values() 
                   if (arg_type is None or arg.argument_type == arg_type) and not arg.is_decision_node]
        sorted_args = sorted(filtered, key=lambda x: x.final_score, reverse=True)
        return sorted_args[:top_k]
    
    def get_decision_nodes(self) -> List[QBAFArgument]:
        """Get the claim node (single in standard QBAF)"""
        return [arg for arg in self.arguments.values() if arg.is_decision_node]
    
    def determine_winner(self, threshold: float = 0.5, yes_option: str = "Yes", no_option: str = "No") -> Tuple[str, float, Dict]:
        """
        Determine the answer based on claim score vs threshold.
        
        Standard QBAF decision:
        - claim_score >= threshold → Yes (claim is TRUE)
        - claim_score < threshold → No (claim is FALSE)
        
        Args:
            threshold: Score threshold for accepting Yes (default 0.5)
            yes_option: Name of the positive option (default "Yes")
            no_option: Name of the negative option (default "No")
        
        Returns:
            Tuple of (winning_option, claim_score, decision_info)
        """
        scores = self.get_option_scores()
        
        # Get the claim node score
        claim_score = scores.get("claim", {}).get("claim_score", 0.5)
        claim_content = scores.get("claim", {}).get("content", "The claim")
        
        # Decision logic: If score >= threshold -> Yes (claim is true), otherwise -> No
        if claim_score >= threshold:
            winner = yes_option
        else:
            winner = no_option
        
        decision_info = {
            "winner": winner,
            "claim_score": claim_score,
            "claim_content": claim_content,
            "threshold": threshold,
            "margin": abs(claim_score - threshold),  # Distance from threshold
            "support_stats": scores.get("support_arguments", {}),
            "attack_stats": scores.get("attack_arguments", {})
        }
        
        return winner, claim_score, decision_info
    
    def _print_graph_statistics(self) -> None:
        """Print statistics about the QBAF graph"""
        total_supports = sum(len(arg.supports) for arg in self.arguments.values())
        total_attacks = sum(len(arg.attacks) for arg in self.arguments.values())
        
        print(f"[QBAF]   Total support relations: {total_supports}")
        print(f"[QBAF]   Total attack relations: {total_attacks}")
        print(f"[QBAF]   Average support per arg: {total_supports / len(self.arguments):.2f}")
        print(f"[QBAF]   Average attack per arg: {total_attacks / len(self.arguments):.2f}")
    
    def export_for_visualization(self) -> Dict:
        """Export graph structure for visualization"""
        nodes = []
        edges = []
        
        for arg_id, arg in self.arguments.items():
            nodes.append({
                "id": arg_id,
                "label": arg.content[:50] + "..." if len(arg.content) > 50 else arg.content,
                "option": arg.parent_option,
                "type": arg.argument_type,
                "base_score": arg.base_score,
                "final_score": arg.final_score,
                "agent_role": arg.agent_role
            })
            
            for supported_id in arg.supports:
                edges.append({
                    "from": arg_id,
                    "to": supported_id,
                    "type": "support",
                    "weight": arg.final_score
                })
            
            for attacked_id in arg.attacks:
                edges.append({
                    "from": arg_id,
                    "to": attacked_id,
                    "type": "attack",
                    "weight": arg.final_score
                })
        
        return {"nodes": nodes, "edges": edges}

    def export_full_graph(self, claim: str = "", task_context: str = "", stage: str = "final") -> Dict:
        """
        Export complete graph state for saving/loading.
        
        This format preserves ALL information needed to reconstruct the graph:
        - Full argument content (not truncated)
        - All scores (base, final, support/attack impact)
        - All relations (supports, attacks, supported_by, attacked_by)
        - Metadata (claim, task context, semantics, stage)
        
        Args:
            claim: The claim text being evaluated
            task_context: Context/description of the task
            stage: "pre_calculation" or "final" to indicate graph state
        
        Returns:
            Dict with complete graph structure
        """
        arguments = []
        for arg_id, arg in self.arguments.items():
            arguments.append({
                "id": arg_id,
                "content": arg.content,  # Full content, not truncated
                "type": arg.argument_type,
                "base_score": arg.base_score,
                "final_score": arg.final_score,
                "support_impact": arg.support_impact,
                "attack_impact": arg.attack_impact,
                "parent_option": arg.parent_option,
                "agent_role": arg.agent_role,
                "agent_name": arg.agent_name,
                "is_decision_node": arg.is_decision_node,
                # Relations
                "supports": list(arg.supports),
                "attacks": list(arg.attacks),
                "supported_by": list(arg.supported_by),
                "attacked_by": list(arg.attacked_by)
            })
        
        return {
            "version": "1.0",
            "stage": stage,  # "pre_calculation" or "final"
            "claim": claim,
            "task_context": task_context,
            "semantics": self.semantics,
            "arguments": arguments,
            "metadata": {
                "total_nodes": len(self.arguments),
                "total_support_relations": sum(len(arg.supports) for arg in self.arguments.values()),
                "total_attack_relations": sum(len(arg.attacks) for arg in self.arguments.values())
            }
        }

    def load_from_graph(self, graph_data: Dict) -> None:
        """
        Load graph state from exported format.
        
        Args:
            graph_data: Dict from export_full_graph
        """
        self.arguments = {}
        self.semantics = graph_data.get("semantics", "df_quad")
        
        for arg_data in graph_data.get("arguments", []):
            arg = QBAFArgument(
                id=arg_data["id"],
                content=arg_data["content"],
                base_score=arg_data.get("base_score", 0.5),
                parent_option=arg_data.get("parent_option"),
                argument_type=arg_data.get("type", "support"),
                agent_role=arg_data.get("agent_role"),
                agent_name=arg_data.get("agent_name"),
                is_decision_node=arg_data.get("is_decision_node", False)
            )
            # Restore computed scores if available
            arg.final_score = arg_data.get("final_score", 0.0)
            arg.support_impact = arg_data.get("support_impact", 0.0)
            arg.attack_impact = arg_data.get("attack_impact", 0.0)
            # Restore relations
            arg.supports = list(arg_data.get("supports", []))
            arg.attacks = list(arg_data.get("attacks", []))
            arg.supported_by = list(arg_data.get("supported_by", []))
            arg.attacked_by = list(arg_data.get("attacked_by", []))
            
            self.arguments[arg.id] = arg

    def save_graph_to_file(self, filepath: str, claim: str = "", task_context: str = "", stage: str = "final") -> str:
        """
        Save graph state to a JSON file.
        
        Args:
            filepath: Path to save the file (e.g., "graph_001.json")
            claim: The claim text
            task_context: Task context description
            stage: "pre_calculation" or "final"
        
        Returns:
            The filepath where the graph was saved
        """
        graph_data = self.export_full_graph(claim=claim, task_context=task_context, stage=stage)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        print(f"[QBAF] Graph saved to {filepath} (stage: {stage})")
        return filepath

    @classmethod
    def load_graph_from_file(cls, filepath: str) -> 'QBAFScorer':
        """
        Load a QBAFScorer from a saved graph file.
        
        Args:
            filepath: Path to the graph JSON file
        
        Returns:
            QBAFScorer instance with loaded graph
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        scorer = cls(semantics=graph_data.get("semantics", "df_quad"))
        scorer.load_from_graph(graph_data)
        return scorer


def apply_qbaf_scoring(
    arguments: List, 
    semantics: str = "quadratic_energy",
    use_semantic_analysis: bool = False,
    use_clash_resolution: bool = False,
    clash_trigger_threshold: float = 0.2,
    task_context: str = "",
    case_text: str = "",
    decision_threshold: float = 0.5,
    claim: str = "The claim is true",
    llm_provider: str = "gemini",
    save_pre_calc_graph: bool = True,
    graph_output_dir: str = "graphs"
) -> Tuple[List, Dict, 'QBAFScorer', bool]:
    """
    Apply standard QBAF scoring to a list of arguments.
    
    Standard QBAF model:
    - Central CLAIM node (base score 0.5)
    - Evidence/Arguments that SUPPORT or ATTACK the claim
    - Inter-argument relations
    - Optional: Clash resolution via LLM (adjusts scores before QBAF computation)
    - claim_score >= threshold → Yes, else No
    
    Args:
        arguments: List of Argument objects (with argument_type="support" or "attack")
        semantics: Type of gradual semantics ("df_quad", "weighted_sum", "weighted_product", "euler_based")
        use_semantic_analysis: Use LLM-based semantic analysis for relations (slower but more accurate)
        use_clash_resolution: Use LLM to resolve clashes between support/attack arguments
        clash_trigger_threshold: Per-pair threshold - only resolve if |arg1_score - arg2_score| < threshold (default 0.2)
        task_context: Description of legal task for semantic analysis context
        case_text: The case/scenario text for clash resolution context
        decision_threshold: Threshold for Yes decision (default 0.5)
        claim: The central thesis being evaluated
        llm_provider: LLM provider for clash resolution ("gpt", "gemini", etc.)
        save_pre_calc_graph: Save graph state before QBAF calculation (default: True)
        graph_output_dir: Directory to save graph files (default: "graphs")
    
    Returns:
        Tuple of (updated_arguments, scores_dict, scorer, used_heuristic_fallback)
        - used_heuristic_fallback: True if any pairs used heuristic instead of LLM analysis
    """
    # Create QBAF scorer
    scorer = QBAFScorer(semantics=semantics)
    
    # Build graph from arguments (standard QBAF - no option_mapping needed)
    used_heuristic_fallback = scorer.build_argument_graph_from_state(
        arguments, 
        option_mapping=None,  # No longer needed in standard QBAF
        use_semantic_analysis=use_semantic_analysis,
        task_context=task_context,
        claim=claim
    )
    
    # Resolve clashes between argument pairs if enabled
    # Each (support, attack) pair is checked individually:
    # - If scores are similar (diff < threshold) → call LLM to resolve
    # - If one is clearly stronger → skip (no LLM call needed)
    if use_clash_resolution and case_text:
        scorer.resolve_argument_clashes(
            arguments=arguments,
            claim=claim,
            case_text=case_text,
            task_context=task_context,
            provider=llm_provider,
            clash_trigger_threshold=clash_trigger_threshold
        )
    
    # === SAVE PRE-CALCULATION GRAPH ===
    # Save the graph state AFTER semantic relations and clash resolution
    # but BEFORE computing final scores
    if save_pre_calc_graph:
        import os
        from datetime import datetime
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_filename = f"graph_{timestamp}_pre_calc.json"
        graph_path = os.path.join(graph_output_dir, graph_filename)
        
        # Ensure directory exists
        os.makedirs(graph_output_dir, exist_ok=True)
        
        # Save pre-calculation state
        scorer.save_graph_to_file(
            filepath=graph_path,
            claim=claim,
            task_context=task_context,
            stage="pre_calculation"
        )
    
    # Compute scores using DF-QuAD (after clash resolution adjustments)
    final_scores = scorer.compute_final_scores()
    
    # Update original arguments with QBAF scores
    for i, arg in enumerate(arguments):
        qbaf_arg = scorer.arguments.get(f"arg_{i}")
        if qbaf_arg:
            # Store both LLM score and QBAF score
            arg.llm_validity_score = arg.validity_score  # Preserve original LLM score
            arg.validity_score = qbaf_arg.final_score    # Replace with QBAF score
            arg.qbaf_support_impact = qbaf_arg.support_impact
            arg.qbaf_attack_impact = qbaf_arg.attack_impact
            arg.clash_adjusted_base = qbaf_arg.base_score  # Track clash-adjusted base score
    
    # Get scores
    scores = scorer.get_option_scores()
    
    # Determine winner
    winner, claim_score, decision_info = scorer.determine_winner(threshold=decision_threshold)
    
    # Print summary
    print("\n=== QBAF Scoring Summary (Standard Model) ===")
    print(f"\nClaim: \"{claim}\"")
    print(f"Clash Resolution: {'ENABLED' if use_clash_resolution else 'DISABLED'}")
    print(f"Claim Score: {claim_score:.3f} ⭐")
    print(f"Threshold: {decision_threshold}")
    print(f"Result: {winner} (claim is {'TRUE' if winner == 'Yes' else 'FALSE'})")
    print(f"Margin from threshold: {decision_info['margin']:.3f}")
    
    # Print argument statistics
    support_stats = scores.get("support_arguments", {})
    attack_stats = scores.get("attack_arguments", {})
    
    print(f"\nSupport Arguments ({support_stats.get('count', 0)}):")
    print(f"  Total Score: {support_stats.get('total_score', 0.0):.3f}")
    print(f"  Average: {support_stats.get('average_score', 0.0):.3f}")
    
    print(f"\nAttack Arguments ({attack_stats.get('count', 0)}):")
    print(f"  Total Score: {attack_stats.get('total_score', 0.0):.3f}")
    print(f"  Average: {attack_stats.get('average_score', 0.0):.3f}")
    
    print(f"\nTop 3 arguments by score:")
    for i, arg in enumerate(scorer.get_top_arguments(top_k=3), 1):
        type_emoji = "🟢" if arg.argument_type == "support" else "🔴"
        print(f"  {i}. {type_emoji} [{arg.argument_type}] Score: {arg.final_score:.3f} - {arg.content[:60]}...")
    
    # Add decision info to scores
    scores['_decision'] = decision_info
    
    # === SAVE FINAL GRAPH ===
    # Save the graph state AFTER computing final scores
    if save_pre_calc_graph:
        import os
        from datetime import datetime
        
        # Use same timestamp as pre-calc graph
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_filename = f"graph_{timestamp}_final.json"
        graph_path = os.path.join(graph_output_dir, graph_filename)
        
        # Save final state
        scorer.save_graph_to_file(
            filepath=graph_path,
            claim=claim,
            task_context=task_context,
            stage="final"
        )
    
    return arguments, scores, scorer, used_heuristic_fallback