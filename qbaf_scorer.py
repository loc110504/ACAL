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
        option_mapping: Dict[str, str],
        use_semantic_analysis: bool = False,
        task_context: str = ""
    ) -> None:
        """
        Build QBAF graph from your existing argument structure.
        
        Two modes:
        1. Heuristic (use_semantic_analysis=False): Simple rules based on options
           - Arguments for Option A attack arguments for Option B
           - Support arguments for same option support each other
        
        2. Semantic (use_semantic_analysis=True): LLM-based NLI analysis
           - Analyzes actual content to determine attack/support/neutral
           - More accurate but slower and uses more API calls
        
        Args:
            arguments: List of Argument objects from your state
            option_mapping: Maps option names to standardized IDs
            use_semantic_analysis: Use LLM-based semantic analysis (default: False for speed)
            task_context: Description of legal task for semantic analysis
        """
        # Clear existing
        self.arguments = {}
        
        # Step 1: Create QBAF arguments
        for i, arg in enumerate(arguments):
            qbaf_arg = QBAFArgument(
                id=f"arg_{i}",
                content=arg.content,
                base_score=arg.validity_score if arg.validity_score is not None else 0.5,
                parent_option=arg.parent_option,
                argument_type=arg.argument_type,
                agent_role=arg.agent_role,
                agent_name=arg.agent_name
            )
            self.add_argument(qbaf_arg)
        
        # Step 2: Infer attack/support relations
        if use_semantic_analysis:
            print(f"[QBAF] Using SEMANTIC ANALYSIS for relation identification...")
            self._build_relations_semantic(arguments, task_context)
        else:
            print(f"[QBAF] Using HEURISTIC RULES for relation identification...")
            self._build_relations_heuristic()
        
        print(f"[QBAF] Built graph with {len(self.arguments)} arguments")
        self._print_graph_statistics()
    
    def _build_relations_heuristic(self) -> None:
        """
        Build relations using simple heuristic rules.
        FAST but less accurate - doesn't consider semantic content.
        """
        arg_list = list(self.arguments.values())
        
        for i, arg_i in enumerate(arg_list):
            for j, arg_j in enumerate(arg_list):
                if i == j:
                    continue
                
                # Rule 1: Arguments for different options attack each other
                if arg_i.parent_option != arg_j.parent_option:
                    self.add_attack(arg_i.id, arg_j.id)
                
                # Rule 2: Arguments of same type for same option support each other
                elif arg_i.parent_option == arg_j.parent_option:
                    if arg_i.argument_type == arg_j.argument_type:
                        self.add_support(arg_i.id, arg_j.id)
    
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
    
    def _compute_df_quad(self) -> Dict[str, float]:
        """
        DF-QuAD (Discontinuity-Free Quantitative Argumentation Debate) semantics.
        Most sophisticated - handles complex attack/support patterns smoothly.
        
        Based on: Rago et al. "Discontinuity-Free Decision Support with Quantitative Argumentation Debates" (KR 2016)
        """
        # Parameters
        alpha = 0.4  # Support weight
        beta = 0.4   # Attack weight
        
        scores = {arg_id: arg.base_score for arg_id, arg in self.arguments.items()}
        
        iteration = 0
        while iteration < self.max_iterations:
            new_scores = {}
            max_change = 0
            
            for arg_id, arg in self.arguments.items():
                # Aggregate support
                if arg.supported_by:
                    support_agg = sum(scores[sup_id] for sup_id in arg.supported_by) / len(arg.supported_by)
                else:
                    support_agg = 0
                
                # Aggregate attack
                if arg.attacked_by:
                    attack_agg = sum(scores[att_id] for att_id in arg.attacked_by) / len(arg.attacked_by)
                else:
                    attack_agg = 0
                
                # DF-QuAD formula: combines base score with support/attack via smooth function
                support_contribution = alpha * support_agg * (1 - arg.base_score)
                attack_reduction = beta * attack_agg * arg.base_score
                
                new_score = arg.base_score + support_contribution - attack_reduction
                new_score = max(0, min(1, new_score))
                
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
        Compute aggregate scores for each option based on argument scores.
        
        IMPORTANT: All arguments are in the same QBAF graph. The DF-QuAD convergence
        already factors in attack/support relations through the graph structure.
        We just SUM all final_scores - no need to differentiate by argument_type.
        
        Returns:
            Dictionary mapping option names to aggregate scores
        """
        option_scores = {}
        
        for arg in self.arguments.values():
            if arg.parent_option not in option_scores:
                option_scores[arg.parent_option] = {
                    "total_score": 0.0,  # Sum of all final_scores (QBAF already handled relations)
                    "support_score": 0.0,  # For analysis only
                    "attack_score": 0.0,   # For analysis only
                    "count": 0
                }
            
            # Add to total score (this is what matters for decision)
            option_scores[arg.parent_option]["total_score"] += arg.final_score
            
            # Track by type for analysis/debugging purposes only
            if arg.argument_type == "support":
                option_scores[arg.parent_option]["support_score"] += arg.final_score
            elif arg.argument_type == "attack":
                option_scores[arg.parent_option]["attack_score"] += arg.final_score
            
            option_scores[arg.parent_option]["count"] += 1
        
        # Compute average score per argument (for fair comparison across options)
        for option, data in option_scores.items():
            data["average_score"] = data["total_score"] / max(data["count"], 1)
        
        return option_scores
    
    def get_top_arguments(self, option: str = None, top_k: int = 5) -> List[QBAFArgument]:
        """Get top-k arguments by final score, optionally filtered by option"""
        filtered = [arg for arg in self.arguments.values() 
                   if option is None or arg.parent_option == option]
        sorted_args = sorted(filtered, key=lambda x: x.final_score, reverse=True)
        return sorted_args[:top_k]
    
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
    task_context: str = ""
) -> Tuple[List, Dict, 'QBAFScorer']:
    """
    Convenience function to apply QBAF scoring to a list of arguments.
    
    Args:
        arguments: List of Argument objects from your state
        semantics: Type of gradual semantics ("weighted_sum", "weighted_product", "euler_based", "df_quad")
        use_semantic_analysis: Use LLM-based semantic analysis for relations (slower but more accurate)
        task_context: Description of legal task for semantic analysis context
    
    Returns:
        Tuple of (updated_arguments, option_scores, scorer)
    """
    # Create QBAF scorer
    scorer = QBAFScorer(semantics=semantics)
    
    # Build graph from arguments
    option_mapping = {}
    for arg in arguments:
        if arg.parent_option not in option_mapping:
            option_mapping[arg.parent_option] = f"opt_{arg.parent_option.lower()}"
    
    scorer.build_argument_graph_from_state(
        arguments, 
        option_mapping,
        use_semantic_analysis=use_semantic_analysis,
        task_context=task_context
    )
    
    # Compute scores
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
    
    # Get option-level scores
    option_scores = scorer.get_option_scores()
    
    # Print summary
    print("\n=== QBAF Scoring Summary ===")
    for option, scores in option_scores.items():
        print(f"\nOption: {option}")
        print(f"  Total QBAF Score: {scores['total_score']:.3f}")
        print(f"  Average per arg:  {scores['average_score']:.3f}")
        print(f"  Breakdown - Support args: {scores['support_score']:.3f}, Attack args: {scores['attack_score']:.3f}")
        print(f"  Argument count: {scores['count']}")
    
    print(f"\nTop 3 arguments overall:")
    for i, arg in enumerate(scorer.get_top_arguments(top_k=3), 1):
        print(f"  {i}. [{arg.parent_option}] Score: {arg.final_score:.3f} - {arg.content[:80]}...")
    
    return arguments, option_scores, scorer