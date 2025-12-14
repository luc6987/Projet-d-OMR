
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve, auc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

@dataclass
class EvalSymbol:
    """Simplified symbol representation for evaluation."""
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    class_probs: np.ndarray  # Probability vector for all classes
    class_id: int  # Argmax class ID (for GT, this is the true class)
    confidence: float = 1.0  # Detection confidence

    @property
    def x1(self) -> float: return self.bbox[0]
    @property
    def y1(self) -> float: return self.bbox[1]
    @property
    def x2(self) -> float: return self.bbox[2]
    @property
    def y2(self) -> float: return self.bbox[3]
    
    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

class MatchAUC:
    """
    Implements the Match+AUC metric.
    
    Metric Definition:
    1. Construct a weighted bipartite graph between Predicted Symbols (V_pred) and Ground Truth Symbols (V_gt).
       Weight w_ij = IoU(b_i, b_j) * p_{i, c_j}
       where p_{i, c_j} is the predicted probability that symbol i belongs to the true class of symbol j.
    2. Solve for Maximum Weight Matching to align V_pred and V_gt.
    3. Filter matches with w_ij < T_match.
    4. Map ground truth edges E_gt to expected edges E_hat on V_pred nodes using the matching.
    5. Compute AUC of the Precision-Recall curve for predicted edges E_pred against E_hat.
    """
    
    def __init__(self, match_threshold: float = 0.05):
        self.match_threshold = match_threshold

    def compute(self, 
                pred_symbols: List[EvalSymbol], 
                pred_edges_scores: Dict[Tuple[int, int], float],
                gt_symbols: List[EvalSymbol],
                gt_edges: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Computes the Match+AUC score.

        Args:
            pred_symbols: List of predicted symbols.
            pred_edges_scores: Dictionary mapping (pred_idx_src, pred_idx_dst) to presence probability [0, 1].
                               This should contain scores for ALL relevant pairs in the graph.
            gt_symbols: List of ground truth symbols.
            gt_edges: List of ground truth edges as (gt_idx_src, gt_idx_dst).
            
        Returns:
            Dictionary containing 'auc', 'precision', 'recall', 'f1', and 'match_score'.
        """
        
        # 1. & 2. Compute Matching
        matching, match_score = self._compute_matching(pred_symbols, gt_symbols)
        
        # Mapping from GT index to Pred index: gt_idx -> pred_idx
        gt_to_pred = {gt_idx: pred_idx for pred_idx, gt_idx in matching}
        
        # 3. Map GT edges to Expected Predicted Edges (E_hat)
        # Only edges where both source and target symbols are successfully matched are considered.
        mapped_gt_edges = set()
        for u_gt, v_gt in gt_edges:
            if u_gt in gt_to_pred and v_gt in gt_to_pred:
                u_pred = gt_to_pred[u_gt]
                v_pred = gt_to_pred[v_gt]
                mapped_gt_edges.add((u_pred, v_pred))
        
        # 4. Compute AUC
        # We evaluate all predicted edges provided in pred_edges_scores.
        # Ground truth labels: 1 if the edge is in mapped_gt_edges, else 0.
        
        y_true = []
        y_scores = []
        
        # Iterate over all possible edges that the model scored
        for (u, v), score in pred_edges_scores.items():
            is_true_edge = 1 if (u, v) in mapped_gt_edges else 0
            y_true.append(is_true_edge)
            y_scores.append(score)
            
        if not y_true:
            return {'auc': 0.0, 'match_score': match_score}

        # Calculate Precision-Recall Curve and AUC
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_score = auc(recall, precision)
        
        return {
            'auc': auc_score,
            'match_score': match_score,
            'num_matched_symbols': len(matching),
            'num_gt_symbols': len(gt_symbols),
            'num_pred_symbols': len(pred_symbols),
            'num_mapped_edges': len(mapped_gt_edges),
            'num_gt_edges': len(gt_edges)
        }

    def _compute_matching(self, pred_symbols: List[EvalSymbol], gt_symbols: List[EvalSymbol]) -> Tuple[List[Tuple[int, int]], float]:
        """
        Computes Maximum Weight Matching.
        Returns list of (pred_idx, gt_idx) tuples and normalized total match score.
        """
        n_pred = len(pred_symbols)
        n_gt = len(gt_symbols)
        
        if n_pred == 0 or n_gt == 0:
            return [], 0.0

        # Construct Weight Matrix [n_pred, n_gt]
        cost_matrix = np.zeros((n_pred, n_gt))
        
        for i, p_sym in enumerate(pred_symbols):
            for j, g_sym in enumerate(gt_symbols):
                iou = compute_iou(p_sym.bbox, g_sym.bbox)
                
                # Check bounds for probability indexing
                if g_sym.class_id < len(p_sym.class_probs):
                    prob = p_sym.class_probs[g_sym.class_id]
                else:
                    # Fallback if dimensions don't match (should handle carefully)
                    prob = 0.0
                
                weight = iou * prob
                
                # We want MAX weight, but scipy does MIN cost. So negate.
                # However, scipy linear_sum_assignment handles linear cost. 
                # To simply use it as max weight matching, we can pass maximize=True in recent SciPy,
                # or negate costs. Let's negate.
                cost_matrix[i, j] = -weight

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        total_weight = 0.0
        
        for r, c in zip(row_ind, col_ind):
            weight = -cost_matrix[r, c]
            if weight >= self.match_threshold:
                matches.append((r, c))
                total_weight += weight
                
        # Metric normalization: Total Weight / max(N_pred, N_gt) ?? 
        # The paper doesn't strictly specify normalization for the *matching* step itself as a metric, 
        # but it helps to track how well we matched.
        normalized_score = total_weight / max(n_pred, n_gt) if max(n_pred, n_gt) > 0 else 0.0
        
        return matches, normalized_score
