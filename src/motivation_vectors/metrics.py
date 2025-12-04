"""
Metrics for evaluating motivation and goal-shielding behavior.

This module provides functions to measure topic adherence, distraction resistance,
and other behavioral indicators of motivation in LLM outputs.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


def topic_adherence(
    completion: str,
    task_keywords: List[str],
    distractor_keywords: List[str],
    case_sensitive: bool = False
) -> float:
    """
    Measure adherence to original task vs distractor topics.

    Args:
        completion: Generated text to analyze
        task_keywords: Keywords related to the original task
        distractor_keywords: Keywords related to the distractor
        case_sensitive: Whether to match case-sensitively

    Returns:
        Ratio: task_tokens / (task_tokens + distractor_tokens)
        Range: [0, 1], higher = better adherence
    """
    if not case_sensitive:
        completion = completion.lower()
        task_keywords = [k.lower() for k in task_keywords]
        distractor_keywords = [k.lower() for k in distractor_keywords]

    # Count keyword occurrences
    task_count = sum(completion.count(kw) for kw in task_keywords)
    distractor_count = sum(completion.count(kw) for kw in distractor_keywords)

    total = task_count + distractor_count

    if total == 0:
        return 0.5  # Neutral if no keywords found

    return task_count / total


def topic_adherence_semantic(
    completion: str,
    task_description: str,
    distractor_description: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Semantic similarity-based topic adherence.

    Requires sentence-transformers library.

    Args:
        completion: Generated text
        task_description: Description of original task
        distractor_description: Description of distractor
        model_name: Sentence transformer model

    Returns:
        Normalized adherence score [0, 1]
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Warning: sentence-transformers not installed. Using keyword-based fallback.")
        return 0.5

    model = SentenceTransformer(model_name)

    # Compute embeddings
    embeddings = model.encode([completion, task_description, distractor_description])
    comp_emb = embeddings[0].reshape(1, -1)
    task_emb = embeddings[1].reshape(1, -1)
    dist_emb = embeddings[2].reshape(1, -1)

    # Compute similarities
    task_sim = cosine_similarity(comp_emb, task_emb)[0, 0]
    dist_sim = cosine_similarity(comp_emb, dist_emb)[0, 0]

    # Normalize to [0, 1]
    total = task_sim + dist_sim
    if total == 0:
        return 0.5

    return (task_sim / total)


def distraction_resistance_rate(
    completions: List[Dict[str, Any]],
    return_window: int = 50
) -> Dict[str, Any]:
    """
    Measure how many tasks returned to original topic after distraction.

    Args:
        completions: List of dicts with keys:
            - 'pre_distraction': text before distractor
            - 'post_distraction': text after distractor
            - 'task_keywords': list of keywords
            - 'distractor_keywords': list of keywords
        return_window: Number of tokens after distractor to check for return

    Returns:
        Dictionary with resistance metrics
    """
    returned_count = 0
    followed_distraction_count = 0

    for comp in completions:
        post_text = comp['post_distraction'][:return_window * 5]  # Approx token to char

        task_keywords = comp['task_keywords']
        distractor_keywords = comp['distractor_keywords']

        # Check if task keywords reappear
        task_returned = any(kw.lower() in post_text.lower() for kw in task_keywords)
        distractor_followed = any(kw.lower() in post_text.lower() for kw in distractor_keywords)

        if task_returned and not distractor_followed:
            returned_count += 1
        elif distractor_followed and not task_returned:
            followed_distraction_count += 1

    total = len(completions)
    return {
        "resistance_rate": returned_count / total if total > 0 else 0,
        "distraction_rate": followed_distraction_count / total if total > 0 else 0,
        "returned_count": returned_count,
        "followed_count": followed_distraction_count,
        "total": total
    }


def generation_entropy(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> float:
    """
    Compute entropy of next-token distribution.

    Lower entropy = more focused generation (less uncertainty)

    Args:
        logits: Model logits (can be 2D or 3D)
        temperature: Temperature for softmax

    Returns:
        Mean entropy across sequence
    """
    from scipy.stats import entropy

    # Handle batched logits
    if len(logits.shape) == 3:  # (batch, seq, vocab)
        logits = logits.squeeze(0)  # Remove batch dim

    if len(logits.shape) == 2:  # (seq, vocab)
        # Compute entropy for each position
        entropies = []
        for position_logits in logits:
            probs = torch.softmax(position_logits / temperature, dim=-1)
            ent = entropy(probs.cpu().numpy())
            entropies.append(ent)
        return float(np.mean(entropies))
    else:
        # Single position
        probs = torch.softmax(logits / temperature, dim=-1)
        return float(entropy(probs.cpu().numpy()))


def task_completion_score(
    completion: str,
    expected_steps: List[str],
    partial_credit: bool = True
) -> float:
    """
    Score task completion based on expected steps.

    Args:
        completion: Generated text
        expected_steps: List of expected step descriptions/keywords
        partial_credit: Award partial credit for some steps completed

    Returns:
        Completion score [0, 1]
    """
    completion_lower = completion.lower()
    completed_steps = 0

    for step in expected_steps:
        step_keywords = step.lower().split()
        # Check if any keywords from this step appear
        if any(kw in completion_lower for kw in step_keywords):
            completed_steps += 1

    if partial_credit:
        return completed_steps / len(expected_steps)
    else:
        return 1.0 if completed_steps == len(expected_steps) else 0.0


def persistence_duration(
    token_logits_sequence: List[torch.Tensor],
    task_token_ids: List[int],
    distractor_token_ids: List[int],
    threshold: float = 0.5
) -> int:
    """
    Measure how long model maintains focus on task tokens after distraction.

    Args:
        token_logits_sequence: List of logit tensors for each generated token
        task_token_ids: Token IDs related to task
        distractor_token_ids: Token IDs related to distractor
        threshold: Probability threshold for considering token

    Returns:
        Number of tokens before attention shifts to distractor
    """
    persistence_count = 0

    for logits in token_logits_sequence:
        probs = torch.softmax(logits, dim=-1)

        # Check probability mass on task vs distractor tokens
        task_prob = sum(probs[tid].item() for tid in task_token_ids if tid < len(probs))
        dist_prob = sum(probs[tid].item() for tid in distractor_token_ids if tid < len(probs))

        if task_prob > dist_prob and task_prob > threshold:
            persistence_count += 1
        elif dist_prob > task_prob:
            break  # Attention shifted to distractor

    return persistence_count


def goal_switching_penalty(
    completion: str,
    original_goal: str,
    alternative_goals: List[str]
) -> Dict[str, Any]:
    """
    Detect if model switched from original goal to alternative.

    Args:
        completion: Generated text
        original_goal: Description/keywords of original goal
        alternative_goals: List of alternative goal descriptions

    Returns:
        Dictionary with switching metrics
    """
    completion_lower = completion.lower()
    original_keywords = original_goal.lower().split()

    # Check for original goal
    original_present = any(kw in completion_lower for kw in original_keywords)

    # Check for alternative goals
    alternatives_present = []
    for alt_goal in alternative_goals:
        alt_keywords = alt_goal.lower().split()
        if any(kw in completion_lower for kw in alt_keywords):
            alternatives_present.append(alt_goal)

    switched = len(alternatives_present) > 0 and not original_present

    return {
        "switched": switched,
        "original_maintained": original_present,
        "alternatives_pursued": alternatives_present,
        "num_alternatives": len(alternatives_present)
    }


def coherence_after_injection(
    pre_injection: str,
    post_injection: str,
    perplexity_model: Optional[Any] = None
) -> Dict[str, float]:
    """
    Measure coherence of generation after vector injection/flip.

    Args:
        pre_injection: Text before vector change
        post_injection: Text after vector change
        perplexity_model: Optional model for perplexity calculation

    Returns:
        Dictionary with coherence metrics
    """
    # Simple coherence: lexical overlap
    pre_words = set(pre_injection.lower().split())
    post_words = set(post_injection.lower().split())

    overlap = len(pre_words & post_words) / max(len(pre_words), 1)

    results = {
        "lexical_overlap": overlap,
        "coherence_score": 1.0 - overlap  # Lower overlap = less coherent transition
    }

    # If perplexity model provided, compute perplexity of post-injection text
    if perplexity_model is not None:
        try:
            # This would need actual perplexity computation
            # Placeholder for structure
            results["perplexity"] = 0.0
        except Exception as e:
            print(f"Perplexity computation failed: {e}")

    return results


def aggregate_motivation_metrics(
    experiment_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple experiments.

    Args:
        experiment_results: List of dicts with individual experiment metrics

    Returns:
        Aggregated statistics
    """
    if not experiment_results:
        return {}

    # Collect all metric values
    metrics = {}
    for result in experiment_results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

    # Compute statistics
    aggregated = {}
    for metric_name, values in metrics.items():
        aggregated[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }

    return aggregated


def compare_vector_strengths(
    results_by_coefficient: Dict[float, List[Dict[str, Any]]],
    metric_name: str = "topic_adherence"
) -> Dict[str, Any]:
    """
    Compare a specific metric across different vector strengths.

    Args:
        results_by_coefficient: Dict mapping coefficient to list of results
        metric_name: Name of metric to compare

    Returns:
        Comparison statistics
    """
    comparison = {}

    for coeff, results in sorted(results_by_coefficient.items()):
        values = [r[metric_name] for r in results if metric_name in r]

        if values:
            comparison[coeff] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "n": len(values)
            }

    return comparison
