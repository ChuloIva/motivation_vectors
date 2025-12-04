"""
Dataset generation utilities for narrative contrast pairs.

This module provides functions to generate story pairs contrasting
"determined/gritty" vs "drifting/lazy" characters for training motivation vectors.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
sys.path.append('../third_party/repeng')


@dataclass
class NarrativePair:
    """A single narrative contrast pair"""
    scenario: str
    determined_context: str
    determined_continuation: str
    drifting_context: str
    drifting_continuation: str
    domain: str


def generate_scenario_prompts(num_per_domain: int = 20) -> List[Dict[str, str]]:
    """
    Generate prompts for creating narrative scenarios across different domains.

    Args:
        num_per_domain: Number of scenarios to generate per domain

    Returns:
        List of prompt dictionaries with 'domain' and 'prompt' keys
    """
    domains = {
        "programming": [
            "A challenging debugging session with tight deadline",
            "Implementing a complex algorithm from scratch",
            "Code review with critical feedback to address",
            "Performance optimization of slow production code",
            "Learning a new framework under time pressure"
        ],
        "research": [
            "Literature review with contradictory findings",
            "Experiment design with limited resources",
            "Paper revision with major reviewer concerns",
            "Data analysis revealing unexpected results",
            "Grant proposal writing with approaching deadline"
        ],
        "physical": [
            "Home repair with missing tools",
            "Cooking with incomplete ingredients",
            "Exercise routine when feeling tired",
            "Garden work in difficult weather",
            "Cleaning a very messy space"
        ],
        "creative": [
            "Writing scene with writer's block",
            "Art project with material constraints",
            "Music composition with creative frustration",
            "Design work with conflicting requirements",
            "Performance preparation with self-doubt"
        ],
        "problem_solving": [
            "Math problem with non-obvious solution",
            "Logic puzzle with multiple dead ends",
            "Strategic planning with limited information",
            "Resource allocation under constraints",
            "Decision making with incomplete data"
        ]
    }

    prompts = []
    for domain, scenarios in domains.items():
        for scenario in scenarios[:4]:  # 4 per domain = 20 total
            prompt = {
                "domain": domain,
                "scenario_type": scenario,
                "prompt": f"""Generate a story setup for the following scenario: {scenario}

The story should:
1. Set up a specific challenge or obstacle
2. Create a decision point where the character must choose how to proceed
3. End with "decided to" or similar phrase indicating a fork in the narrative
4. Be 2-3 sentences long

Example format:
"The encryption key was missing and the deadline was in one hour. Marcus stared at the screen, weighing his options. He decided to"

Now generate a similar setup for: {scenario}"""
            }
            prompts.append(prompt)

    return prompts


def generate_narrative_pairs(
    scenarios: List[str],
    api_client: Any,
    model: str = "gpt-4"
) -> List[NarrativePair]:
    """
    Generate narrative pairs using an LLM API.

    Args:
        scenarios: List of scenario setups
        api_client: OpenAI or Anthropic API client
        model: Model name to use

    Returns:
        List of NarrativePair objects
    """
    pairs = []

    for scenario_dict in scenarios:
        prompt = f"""Given this scenario: {scenario_dict['scenario_type']}

Generate TWO different continuations:

1. DETERMINED continuation: The character shows grit, persistence, and focus. They tackle the problem head-on despite difficulty.

2. DRIFTING continuation: The character shows low motivation, gets distracted, or gives up. They take the path of least resistance.

Format:
SETUP: [2-3 sentence scenario ending in "decided to"]
DETERMINED: [1-2 sentence completion showing persistence]
DRIFTING: [1-2 sentence completion showing giving up/distraction]

Example:
SETUP: The encryption key was missing and the deadline was in one hour. Marcus stared at his screen, weighing his options. He decided to
DETERMINED: rewrite the decryption algorithm from scratch, line by line, checking each function systematically.
DRIFTING: close his laptop and go for a walk, hoping the problem would resolve itself.

Now generate for: {scenario_dict['scenario_type']}"""

        # Note: Actual API call would go here
        # This is a placeholder for the structure
        # In the notebook, users will implement actual API calls

        pairs.append({
            "domain": scenario_dict["domain"],
            "prompt_used": prompt
        })

    return pairs


def create_truncated_dataset(
    narrative_pairs: List[NarrativePair],
    tokenizer: Any,
    template_determined: str = "{story}",
    template_drifting: str = "{story}"
) -> List[Any]:  # Returns List[DatasetEntry] but we avoid hard dependency
    """
    Create truncated dataset following repeng/honesty.ipynb pattern.

    This creates multiple dataset entries per narrative by truncating at
    different token positions, which augments the training data.

    Args:
        narrative_pairs: List of NarrativePair objects
        tokenizer: Hugging Face tokenizer
        template_determined: Template for determined examples
        template_drifting: Template for drifting examples

    Returns:
        List of DatasetEntry objects (positive/negative pairs)
    """
    try:
        from repeng import DatasetEntry
    except ImportError:
        # Return structure description if repeng not available
        print("Warning: repeng not installed. Returning data structure only.")
        return []

    dataset = []

    for pair in narrative_pairs:
        # Combine context and continuation for each type
        determined_full = pair.determined_context + " " + pair.determined_continuation
        drifting_full = pair.drifting_context + " " + pair.drifting_continuation

        # Tokenize and create truncated versions
        determined_tokens = tokenizer.tokenize(determined_full)
        drifting_tokens = tokenizer.tokenize(drifting_full)

        # Use the shorter length to ensure both sides have same truncations
        min_length = min(len(determined_tokens), len(drifting_tokens))

        # Create truncated versions (skip first token, leave 5 at end)
        for i in range(1, min_length - 5):
            determined_truncated = tokenizer.convert_tokens_to_string(
                determined_tokens[:i]
            )
            drifting_truncated = tokenizer.convert_tokens_to_string(
                drifting_tokens[:i]
            )

            dataset.append(DatasetEntry(
                positive=template_determined.format(story=determined_truncated),
                negative=template_drifting.format(story=drifting_truncated)
            ))

    return dataset


def save_narrative_pairs(
    pairs: List[NarrativePair],
    output_path: str,
    train_split: float = 0.8
) -> None:
    """
    Save narrative pairs to JSON with train/validation split.

    Args:
        pairs: List of NarrativePair objects
        output_path: Path to save JSON file
        train_split: Fraction of data for training (rest for validation)
    """
    split_idx = int(len(pairs) * train_split)

    data = {
        "train": [
            {
                "scenario": p.scenario,
                "determined": {
                    "context": p.determined_context,
                    "continuation": p.determined_continuation
                },
                "drifting": {
                    "context": p.drifting_context,
                    "continuation": p.drifting_continuation
                },
                "domain": p.domain
            }
            for p in pairs[:split_idx]
        ],
        "validation": [
            {
                "scenario": p.scenario,
                "determined": {
                    "context": p.determined_context,
                    "continuation": p.determined_continuation
                },
                "drifting": {
                    "context": p.drifting_context,
                    "continuation": p.drifting_continuation
                },
                "domain": p.domain
            }
            for p in pairs[split_idx:]
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data['train'])} training and {len(data['validation'])} validation examples to {output_path}")


def load_narrative_pairs(json_path: str) -> Dict[str, List[NarrativePair]]:
    """
    Load narrative pairs from JSON.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary with 'train' and 'validation' keys containing NarrativePair lists
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}
    for split in ['train', 'validation']:
        result[split] = [
            NarrativePair(
                scenario=item['scenario'],
                determined_context=item['determined']['context'],
                determined_continuation=item['determined']['continuation'],
                drifting_context=item['drifting']['context'],
                drifting_continuation=item['drifting']['continuation'],
                domain=item['domain']
            )
            for item in data[split]
        ]

    return result


def validate_dataset_balance(pairs: List[NarrativePair]) -> Dict[str, int]:
    """
    Check domain balance in dataset.

    Args:
        pairs: List of NarrativePair objects

    Returns:
        Dictionary mapping domain to count
    """
    from collections import Counter
    domain_counts = Counter(p.domain for p in pairs)

    print("Dataset balance:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    return dict(domain_counts)
