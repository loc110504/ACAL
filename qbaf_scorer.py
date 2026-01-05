import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import math
from semantic_relation_analyzer import analyze_argument_relations_semantic


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
    """
    
    def __init__(self, semantics: str = "df_quad", convergence_threshold: float = 0.001, max_iterations: int = 100):
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
    ) -> None:
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
        if use_semantic_analysis:
            print(f"[QBAF] Using SEMANTIC ANALYSIS for inter-argument relations...")
            self._build_relations_semantic(arguments, task_context)
        else:
            print(f"[QBAF] Using HEURISTIC RULES for inter-argument relations...")
            self._build_relations_heuristic()
        
        print(f"[QBAF] Built graph with {len(self.arguments)} nodes (1 claim + {len(arguments)} arguments)")
        self._print_graph_statistics()
    
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
    
    def _build_relations_semantic(self, arguments: List, task_context: str) -> None:
        """
        Build relations using LLM-based semantic analysis.
        SLOW but more accurate - analyzes actual argument content.
        """
        # Analyze all relations using semantic analyzer
        relations = analyze_argument_relations_semantic(
            arguments=arguments,
            task_context=task_context,
            use_semantic=True,
            threshold_attack=0.6,
            threshold_support=0.6
        )
        
        # Add relations to QBAF graph
        for (src_id, tgt_id), relation in relations.items():
            if relation.relation_type == "attack":
                self.add_attack(src_id, tgt_id)
            elif relation.relation_type == "support":
                self.add_support(src_id, tgt_id)
            # neutral relations are not added
    
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
        
        # Decision logic: If score >= threshold → Yes (claim is true), otherwise → No
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


def apply_qbaf_scoring(
    arguments: List, 
    semantics: str = "df_quad",
    use_semantic_analysis: bool = False,
    task_context: str = "",
    decision_threshold: float = 0.5,
    claim: str = "The claim is true"
) -> Tuple[List, Dict, 'QBAFScorer']:
    """
    Apply standard QBAF scoring to a list of arguments.
    
    Standard QBAF model:
    - Central CLAIM node (base score 0.5)
    - Evidence/Arguments that SUPPORT or ATTACK the claim
    - Inter-argument relations
    - claim_score >= threshold → Yes, else No
    
    Args:
        arguments: List of Argument objects (with argument_type="support" or "attack")
        semantics: Type of gradual semantics ("df_quad", "weighted_sum", "weighted_product", "euler_based")
        use_semantic_analysis: Use LLM-based semantic analysis for relations (slower but more accurate)
        task_context: Description of legal task for semantic analysis context
        decision_threshold: Threshold for Yes decision (default 0.5)
        claim: The central thesis being evaluated
    
    Returns:
        Tuple of (updated_arguments, scores_dict, scorer)
    """
    # Create QBAF scorer
    scorer = QBAFScorer(semantics=semantics)
    
    # Build graph from arguments (standard QBAF - no option_mapping needed)
    scorer.build_argument_graph_from_state(
        arguments, 
        option_mapping=None,  # No longer needed in standard QBAF
        use_semantic_analysis=use_semantic_analysis,
        task_context=task_context,
        claim=claim
    )
    
    # Compute scores using DF-QuAD
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
    
    # Get scores
    scores = scorer.get_option_scores()
    
    # Determine winner
    winner, claim_score, decision_info = scorer.determine_winner(threshold=decision_threshold)
    
    # Print summary
    print("\n=== QBAF Scoring Summary (Standard Model) ===")
    print(f"\nClaim: \"{claim}\"")
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
    
    return arguments, scores, scorer