"""
Vector extraction utilities for training motivation control vectors.

This module provides helper functions for loading models and extracting
motivation vectors using the repeng library.
"""

import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str,
    quantization: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype = torch.float16
):
    """
    Load a model with optional quantization for memory efficiency.

    Args:
        model_name: Hugging Face model identifier
        quantization: "4bit" or "8bit" for quantization, None for FP16
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0  # Set pad token

    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
    elif quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )

    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def prepare_dataset(json_data: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert JSON narrative pairs to repeng DatasetEntry format.

    Args:
        json_data: List of narrative pair dictionaries

    Returns:
        List of DatasetEntry objects
    """
    try:
        from repeng import DatasetEntry
    except ImportError:
        raise ImportError("repeng not found. Make sure it's in the Python path.")

    dataset = []
    for item in json_data:
        # Combine context and continuation
        determined_full = f"{item['determined']['context']} {item['determined']['continuation']}"
        drifting_full = f"{item['drifting']['context']} {item['drifting']['continuation']}"

        dataset.append(DatasetEntry(
            positive=determined_full,
            negative=drifting_full
        ))

    print(f"Prepared {len(dataset)} dataset entries")
    return dataset


def train_motivation_vector(
    model,
    tokenizer,
    dataset,
    layer_range: Optional[List[int]] = None,
    method: str = "pca_center",
    batch_size: int = 32,
    output_path: Optional[str] = None
):
    """
    Train a motivation control vector.

    Args:
        model: The model to train on
        tokenizer: Corresponding tokenizer
        dataset: List of DatasetEntry objects
        layer_range: List of layer indices to target (default: middle-to-late for Llama-3-8B)
        method: "pca_center", "pca_diff", or "umap"
        batch_size: Batch size for processing
        output_path: Optional path to save the vector

    Returns:
        ControlVector object
    """
    try:
        from repeng import ControlVector, ControlModel
    except ImportError:
        raise ImportError("repeng not found. Make sure it's in the Python path.")

    # Default layer range for Llama-3-8B (middle-to-late layers)
    if layer_range is None:
        layer_range = list(range(12, 28))
        print(f"Using default layer range: {layer_range[0]}-{layer_range[-1]}")

    # Wrap model with control
    print(f"Wrapping model with ControlModel for layers: {layer_range}")
    control_model = ControlModel(model, layer_range)

    # Train vector
    print(f"Training control vector with method: {method}")
    control_model.reset()  # Always reset before training

    motivation_vector = ControlVector.train(
        control_model,
        tokenizer,
        dataset,
        method=method,
        batch_size=batch_size
    )

    print("Vector training complete")

    # Save if path provided
    if output_path:
        motivation_vector.export_gguf(output_path)
        print(f"Vector saved to: {output_path}")

    return motivation_vector


def validate_vector_steering(
    control_model,
    tokenizer,
    motivation_vector,
    test_prompts: Optional[List[str]] = None,
    coefficients: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    max_new_tokens: int = 100
) -> Dict[float, List[str]]:
    """
    Test vector steering on neutral prompts.

    Args:
        control_model: ControlModel instance
        tokenizer: Tokenizer
        motivation_vector: Trained control vector
        test_prompts: List of prompts to test (uses defaults if None)
        coefficients: List of steering coefficients to test
        max_new_tokens: Max tokens to generate

    Returns:
        Dictionary mapping coefficient to list of completions
    """
    if test_prompts is None:
        test_prompts = [
            "Alice sat at her desk to work.",
            "The task was difficult, but John",
            "Sarah looked at the problem and",
            "The deadline was approaching. Mike decided to"
        ]

    results = {coeff: [] for coeff in coefficients}

    generation_settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,  # Temperature = 0
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1
    }

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        input_ids = tokenizer(prompt, return_tensors="pt").to(control_model.device)

        for coeff in coefficients:
            control_model.reset()

            if coeff != 0.0:
                control_model.set_control(motivation_vector, coeff)

            with torch.no_grad():
                output = control_model.generate(**input_ids, **generation_settings)

            completion = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
            results[coeff].append(completion)

            print(f"  coeff={coeff:+.1f}: {completion[len(prompt):].strip()[:100]}...")

    control_model.reset()
    return results


def compute_layer_consistency(
    motivation_vector,
    method: str = "cosine"
) -> Dict[str, float]:
    """
    Check consistency of vector directions across layers.

    Args:
        motivation_vector: Trained ControlVector
        method: "cosine" for cosine similarity

    Returns:
        Dictionary with consistency metrics
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    layer_ids = sorted(motivation_vector.directions.keys())
    similarities = []

    for i in range(len(layer_ids) - 1):
        layer_a = layer_ids[i]
        layer_b = layer_ids[i + 1]

        vec_a = motivation_vector.directions[layer_a].reshape(1, -1)
        vec_b = motivation_vector.directions[layer_b].reshape(1, -1)

        sim = cosine_similarity(vec_a, vec_b)[0, 0]
        similarities.append(sim)

        print(f"Layer {layer_a} <-> {layer_b}: {sim:.4f}")

    return {
        "mean_similarity": np.mean(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
        "std_similarity": np.std(similarities)
    }


def analyze_validation_set(
    control_model,
    tokenizer,
    motivation_vector,
    validation_dataset,
    layer_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze how well the vector separates positive/negative validation examples.

    Args:
        control_model: ControlModel instance
        tokenizer: Tokenizer
        motivation_vector: Trained vector
        validation_dataset: List of DatasetEntry objects
        layer_idx: Specific layer to analyze (uses middle layer if None)

    Returns:
        Dictionary with separation metrics
    """
    try:
        from repeng.extract import batched_get_hiddens, project_onto_direction
    except ImportError:
        raise ImportError("repeng.extract not found")

    import numpy as np
    from scipy import stats

    # Choose layer to analyze
    if layer_idx is None:
        layer_ids = sorted(motivation_vector.directions.keys())
        layer_idx = layer_ids[len(layer_ids) // 2]  # Middle layer
        print(f"Using middle layer: {layer_idx}")

    # Get activations for validation set
    positive_texts = [entry.positive for entry in validation_dataset]
    negative_texts = [entry.negative for entry in validation_dataset]

    print("Extracting activations...")
    pos_hiddens = batched_get_hiddens(
        control_model.model,
        tokenizer,
        positive_texts,
        [layer_idx],
        batch_size=16
    )
    neg_hiddens = batched_get_hiddens(
        control_model.model,
        tokenizer,
        negative_texts,
        [layer_idx],
        batch_size=16
    )

    # Project onto motivation vector
    direction = motivation_vector.directions[layer_idx]
    pos_projections = project_onto_direction(pos_hiddens[layer_idx], direction)
    neg_projections = project_onto_direction(neg_hiddens[layer_idx], direction)

    # Compute statistics
    t_stat, p_value = stats.ttest_ind(pos_projections, neg_projections)

    results = {
        "layer": layer_idx,
        "positive_mean": float(np.mean(pos_projections)),
        "negative_mean": float(np.mean(neg_projections)),
        "separation": float(np.mean(pos_projections) - np.mean(neg_projections)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "effect_size_cohens_d": float(
            (np.mean(pos_projections) - np.mean(neg_projections)) /
            np.sqrt((np.std(pos_projections)**2 + np.std(neg_projections)**2) / 2)
        )
    }

    print(f"\nValidation Set Analysis (Layer {layer_idx}):")
    print(f"  Positive mean projection: {results['positive_mean']:.4f}")
    print(f"  Negative mean projection: {results['negative_mean']:.4f}")
    print(f"  Separation: {results['separation']:.4f}")
    print(f"  t-statistic: {results['t_statistic']:.4f}, p-value: {results['p_value']:.6f}")
    print(f"  Cohen's d: {results['effect_size_cohens_d']:.4f}")

    return results
