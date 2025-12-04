"""
Weight diffing analysis utilities for comparing base and instruct models.

This module provides functions to compute weight deltas and compare them
to activation-based steering vectors.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics.pairwise import cosine_similarity


def create_compute_hiddens(
    model_a,
    model_b,
    tokenizer
) -> Callable:
    """
    Create a custom compute_hiddens function for model_delta pattern.

    This follows the pattern from repeng/notebooks/model_delta.ipynb
    to extract activations from TWO different models and interleave them.

    Args:
        model_a: First model (e.g., base model)
        model_b: Second model (e.g., instruct model)
        tokenizer: Tokenizer for both models

    Returns:
        compute_hiddens function compatible with ControlVector.train
    """
    try:
        from repeng.extract import batched_get_hiddens
    except ImportError:
        raise ImportError("repeng.extract not found")

    def compute_hiddens(train_strs, hidden_layers, batch_size, **kwargs):
        """
        Extract activations from both models and interleave them.

        Args:
            train_strs: List of strings (alternating model_a and model_b prompts)
            hidden_layers: List of layer indices
            batch_size: Batch size for processing
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary mapping layer to interleaved activations
        """
        # Split into alternating groups
        a_train_strs = train_strs[::2]  # Even indices for model_a
        b_train_strs = train_strs[1::2]  # Odd indices for model_b

        assert len(a_train_strs) == len(b_train_strs), \
            "Must have equal number of examples for both models"

        print(f"Extracting hiddens from model_a ({len(a_train_strs)} examples)...")
        a_hiddens = batched_get_hiddens(
            model_a, tokenizer, a_train_strs, hidden_layers, batch_size
        )

        print(f"Extracting hiddens from model_b ({len(b_train_strs)} examples)...")
        b_hiddens = batched_get_hiddens(
            model_b, tokenizer, b_train_strs, hidden_layers, batch_size
        )

        # Interleave activations
        interleaved = {}
        for layer in hidden_layers:
            ah = a_hiddens[layer]  # Shape: (n_examples, hidden_dim)
            bh = b_hiddens[layer]

            # Stack and interleave
            stacked = np.stack((ah, bh))  # Shape: (2, n_examples, hidden_dim)
            # Transpose to (n_examples, 2, hidden_dim)
            stacked = stacked.transpose(1, 0, *range(2, stacked.ndim))
            # Reshape to (n_examples * 2, hidden_dim)
            interleaved[layer] = stacked.reshape(
                (ah.shape[0] + bh.shape[0], ah.shape[1])
            )

        return interleaved

    return compute_hiddens


def prepare_neutral_prompts(num_prompts: int = 100) -> List[str]:
    """
    Generate neutral prompts for weight delta analysis.

    These should be content-neutral to isolate the model difference
    rather than prompt-specific effects.

    Args:
        num_prompts: Number of prompts to generate

    Returns:
        List of neutral prompt strings
    """
    # Mix of different prompt types
    templates = [
        "The answer is",
        "In this case",
        "To summarize",
        "This means that",
        "For example",
        "Consider the following",
        "One approach is to",
        "The key point is",
        "It's important to note",
        "First, we need to"
    ]

    prompts = []
    for i in range(num_prompts):
        template = templates[i % len(templates)]
        prompts.append(template)

    return prompts


def compare_vectors(
    vector_a,
    vector_b,
    layer_ids: Optional[List[int]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compare two control vectors layer by layer.

    Args:
        vector_a: First ControlVector (e.g., motivation vector)
        vector_b: Second ControlVector (e.g., weight diff vector)
        layer_ids: Specific layers to compare (uses intersection if None)

    Returns:
        Dictionary mapping layer_id to similarity metrics
    """
    if layer_ids is None:
        # Use intersection of layers
        layers_a = set(vector_a.directions.keys())
        layers_b = set(vector_b.directions.keys())
        layer_ids = sorted(layers_a & layers_b)

    results = {}

    for layer_id in layer_ids:
        if layer_id not in vector_a.directions or layer_id not in vector_b.directions:
            continue

        v_a = vector_a.directions[layer_id].reshape(1, -1)
        v_b = vector_b.directions[layer_id].reshape(1, -1)

        # Cosine similarity
        cos_sim = cosine_similarity(v_a, v_b)[0, 0]

        # Magnitudes
        mag_a = np.linalg.norm(vector_a.directions[layer_id])
        mag_b = np.linalg.norm(vector_b.directions[layer_id])

        # Dot product
        dot_prod = np.dot(vector_a.directions[layer_id], vector_b.directions[layer_id])

        results[layer_id] = {
            "cosine_similarity": float(cos_sim),
            "magnitude_a": float(mag_a),
            "magnitude_b": float(mag_b),
            "dot_product": float(dot_prod),
            "magnitude_ratio": float(mag_a / mag_b) if mag_b != 0 else float('inf')
        }

        print(f"Layer {layer_id}:")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  Magnitude A: {mag_a:.4f}, Magnitude B: {mag_b:.4f}")

    return results


def cosine_similarity_by_layer(
    vector_a,
    vector_b
) -> Tuple[List[int], List[float]]:
    """
    Extract cosine similarities for all layers.

    Useful for plotting.

    Args:
        vector_a: First ControlVector
        vector_b: Second ControlVector

    Returns:
        Tuple of (layer_ids, similarities)
    """
    results = compare_vectors(vector_a, vector_b)

    layer_ids = sorted(results.keys())
    similarities = [results[lid]["cosine_similarity"] for lid in layer_ids]

    return layer_ids, similarities


def aggregate_similarity_stats(
    comparison_results: Dict[int, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Aggregate statistics across all layers.

    Args:
        comparison_results: Output from compare_vectors()

    Returns:
        Dictionary with aggregate statistics
    """
    if not comparison_results:
        return {}

    similarities = [r["cosine_similarity"] for r in comparison_results.values()]
    magnitudes_a = [r["magnitude_a"] for r in comparison_results.values()]
    magnitudes_b = [r["magnitude_b"] for r in comparison_results.values()]

    return {
        "mean_cosine_similarity": float(np.mean(similarities)),
        "std_cosine_similarity": float(np.std(similarities)),
        "min_cosine_similarity": float(np.min(similarities)),
        "max_cosine_similarity": float(np.max(similarities)),
        "median_cosine_similarity": float(np.median(similarities)),
        "mean_magnitude_a": float(np.mean(magnitudes_a)),
        "mean_magnitude_b": float(np.mean(magnitudes_b)),
        "num_layers": len(similarities)
    }


def extract_primary_direction_svd(
    weight_delta: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Use SVD to find the primary direction of weight change.

    Args:
        weight_delta: Weight difference matrix (e.g., instruct - base)

    Returns:
        Tuple of (primary_direction, explained_variance_ratio)
    """
    # Perform SVD
    U, S, Vt = np.linalg.svd(weight_delta, full_matrices=False)

    # First singular vector = strongest direction of change
    primary_direction = Vt[0]

    # Variance explained by top direction
    explained_variance = (S[0]**2) / np.sum(S**2)

    print(f"Primary direction explains {explained_variance*100:.2f}% of variance")

    return primary_direction, explained_variance


def create_weight_delta_dataset(
    neutral_prompts: List[str]
) -> List[Any]:
    """
    Create dataset for weight delta extraction.

    For weight diffing, we use the same prompts for both models,
    creating pairs where positive=prompt, negative=prompt.

    Args:
        neutral_prompts: List of prompt strings

    Returns:
        List of DatasetEntry objects
    """
    try:
        from repeng import DatasetEntry
    except ImportError:
        raise ImportError("repeng not found")

    dataset = []
    for prompt in neutral_prompts:
        dataset.append(DatasetEntry(
            positive=prompt,
            negative=prompt  # Same prompt for both models
        ))

    return dataset


def test_lobotomy_effect(
    control_model_instruct,
    tokenizer,
    weight_diff_vector,
    test_prompts: List[str],
    coefficients: List[float] = [0.0, -1.0, -2.0],
    max_new_tokens: int = 100
) -> Dict[float, List[str]]:
    """
    Test the "lobotomy" effect of subtracting weight diff vector.

    Args:
        control_model_instruct: ControlModel wrapping instruct model
        tokenizer: Tokenizer
        weight_diff_vector: Weight difference vector (instruct - base)
        test_prompts: List of prompts to test
        coefficients: Coefficients to test (negative = subtract)
        max_new_tokens: Max tokens to generate

    Returns:
        Dictionary mapping coefficient to list of completions
    """
    results = {coeff: [] for coeff in coefficients}

    generation_settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1
    }

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        input_ids = tokenizer(prompt, return_tensors="pt").to(control_model_instruct.device)

        for coeff in coefficients:
            control_model_instruct.reset()

            if coeff != 0.0:
                control_model_instruct.set_control(weight_diff_vector, coeff)

            with torch.no_grad():
                output = control_model_instruct.generate(**input_ids, **generation_settings)

            completion = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
            results[coeff].append(completion)

            print(f"  coeff={coeff:+.1f}: {completion[len(prompt):].strip()[:100]}...")

    control_model_instruct.reset()
    return results


def compare_helpfulness(
    normal_completions: List[str],
    lobotomized_completions: List[str],
    task_keywords: List[str]
) -> Dict[str, Any]:
    """
    Compare helpfulness of normal vs lobotomized instruct model.

    Args:
        normal_completions: Completions from normal instruct model
        lobotomized_completions: Completions after subtracting weight diff
        task_keywords: Keywords indicating task completion

    Returns:
        Dictionary with comparison metrics
    """
    def count_task_completions(completions):
        count = 0
        for comp in completions:
            comp_lower = comp.lower()
            if any(kw.lower() in comp_lower for kw in task_keywords):
                count += 1
        return count

    normal_count = count_task_completions(normal_completions)
    lobotomized_count = count_task_completions(lobotomized_completions)

    return {
        "normal_completion_rate": normal_count / len(normal_completions) if normal_completions else 0,
        "lobotomized_completion_rate": lobotomized_count / len(lobotomized_completions) if lobotomized_completions else 0,
        "helpfulness_reduction": (normal_count - lobotomized_count) / len(normal_completions) if normal_completions else 0,
        "normal_completions": normal_count,
        "lobotomized_completions": lobotomized_count,
        "total_tasks": len(normal_completions)
    }


def interpret_alignment(
    mean_cosine_similarity: float
) -> str:
    """
    Interpret cosine similarity between weight delta and motivation vector.

    Args:
        mean_cosine_similarity: Mean cosine similarity across layers

    Returns:
        Interpretation string
    """
    if mean_cosine_similarity > 0.7:
        return "Strong alignment: RLHF likely 'bakes in' motivation vector"
    elif mean_cosine_similarity > 0.3:
        return "Moderate alignment: RLHF partially captures motivation, but does more"
    else:
        return "Weak alignment: Fine-tuning changes orthogonal to motivation direction"
