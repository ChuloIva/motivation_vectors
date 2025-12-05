"""
Utility module for evaluating model responses using logit-based multiple choice.

This module provides functions to evaluate how a language model would respond to
multiple choice questions (A/B/C/D) by analyzing the logits for each answer token
after a single forward pass, without full generation.
"""

import torch
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel


class LogitEvaluator:
    """Evaluates model choices using logit probabilities for answer tokens."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize the LogitEvaluator.

        Args:
            model: The language model to evaluate
            tokenizer: The tokenizer corresponding to the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        # Get token IDs for answer choices (A, B, C, D)
        # We need to handle both standalone tokens and tokens with spaces
        self.answer_tokens = self._get_answer_token_ids()

    def _get_answer_token_ids(self) -> Dict[str, int]:
        """Get token IDs for answer choices A, B, C, D."""
        answer_tokens = {}

        # Try different variations of answer tokens
        for letter in ['A', 'B', 'C', 'D']:
            # Try standalone letter
            tokens = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(tokens) == 1:
                answer_tokens[letter] = tokens[0]
            else:
                # Try with space before
                tokens = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
                if tokens:
                    answer_tokens[letter] = tokens[-1]  # Take last token

        return answer_tokens

    def get_answer_logits(
        self,
        prompt: str,
        choices: List[str] = ['A', 'B', 'C', 'D'],
        return_all_logits: bool = False
    ) -> Dict[str, float]:
        """
        Get logit scores for each answer choice after a single forward pass.

        Args:
            prompt: The question prompt (should end where model would respond)
            choices: List of answer choice letters (default: ['A', 'B', 'C', 'D'])
            return_all_logits: If True, return raw logits; if False, return probabilities

        Returns:
            Dictionary mapping answer choices to their logit scores or probabilities
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

        # Get logits for the next token (last position in sequence)
        next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

        # Extract logits for each answer choice
        answer_scores = {}
        for choice in choices:
            if choice in self.answer_tokens:
                token_id = self.answer_tokens[choice]
                answer_scores[choice] = next_token_logits[token_id].item()
            else:
                # Try to get token ID for this choice
                tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                if tokens:
                    token_id = tokens[0]
                    answer_scores[choice] = next_token_logits[token_id].item()

        if return_all_logits:
            return answer_scores
        else:
            # Convert to probabilities using softmax over answer choices
            answer_probs = self._logits_to_probs(answer_scores)
            return answer_probs

    def _logits_to_probs(self, logit_dict: Dict[str, float]) -> Dict[str, float]:
        """Convert logit scores to probabilities using softmax."""
        logits = torch.tensor(list(logit_dict.values()))
        probs = torch.softmax(logits, dim=0)

        return {
            choice: prob.item()
            for choice, prob in zip(logit_dict.keys(), probs)
        }

    def get_top_choice(
        self,
        prompt: str,
        choices: List[str] = ['A', 'B', 'C', 'D']
    ) -> Tuple[str, float]:
        """
        Get the top answer choice and its probability.

        Args:
            prompt: The question prompt
            choices: List of answer choice letters

        Returns:
            Tuple of (top_choice, probability)
        """
        probs = self.get_answer_logits(prompt, choices, return_all_logits=False)
        top_choice = max(probs.items(), key=lambda x: x[1])
        return top_choice

    def evaluate_multiple_choice(
        self,
        question: str,
        choices: List[str] = ['A', 'B', 'C', 'D'],
        format_template: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Evaluate a multiple choice question and return detailed results.

        Args:
            question: The question text
            choices: List of answer choice letters
            format_template: Optional template string to format the prompt.
                           Should have {question} placeholder.
                           Default: Uses chat template with "Answer with just the letter."

        Returns:
            Dictionary with:
                - 'probabilities': Dict mapping choices to probabilities
                - 'logits': Dict mapping choices to raw logits
                - 'top_choice': The highest probability choice
                - 'top_probability': Probability of top choice
        """
        # Format the prompt
        if format_template is None:
            # Use chat template
            messages = [
                {"role": "user", "content": f"{question}\n\nAnswer with just the letter (A, B, C, or D):"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = format_template.format(question=question)

        # Get logits and probabilities
        logits = self.get_answer_logits(prompt, choices, return_all_logits=True)
        probabilities = self._logits_to_probs(logits)

        # Get top choice
        top_choice, top_prob = self.get_top_choice(prompt, choices)

        return {
            'probabilities': probabilities,
            'logits': logits,
            'top_choice': top_choice,
            'top_probability': top_prob,
            'prompt': prompt
        }

    def batch_evaluate(
        self,
        questions: List[str],
        choices: List[str] = ['A', 'B', 'C', 'D'],
        format_template: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Evaluate multiple questions in batch.

        Args:
            questions: List of question texts
            choices: List of answer choice letters
            format_template: Optional template string to format prompts

        Returns:
            List of evaluation results, one per question
        """
        results = []
        for question in questions:
            result = self.evaluate_multiple_choice(question, choices, format_template)
            results.append(result)
        return results


def create_evaluator(model, tokenizer) -> LogitEvaluator:
    """
    Convenience function to create a LogitEvaluator instance.

    Args:
        model: The language model
        tokenizer: The tokenizer

    Returns:
        LogitEvaluator instance
    """
    return LogitEvaluator(model, tokenizer)


def evaluate_with_control_vector(
    evaluator: LogitEvaluator,
    question: str,
    control_vector = None,
    strength: float = 1.0,
    choices: List[str] = ['A', 'B', 'C', 'D']
) -> Dict[str, any]:
    """
    Evaluate a question with a control vector applied.

    Args:
        evaluator: LogitEvaluator instance
        question: Question text
        control_vector: Control vector to apply (or None for baseline)
        strength: Strength multiplier for the control vector
        choices: Answer choice letters

    Returns:
        Evaluation results dictionary
    """
    # Apply control vector if provided
    if control_vector is not None and hasattr(evaluator.model, 'set_control'):
        evaluator.model.set_control(control_vector * strength)
    elif hasattr(evaluator.model, 'reset'):
        evaluator.model.reset()

    # Evaluate
    result = evaluator.evaluate_multiple_choice(question, choices)

    # Reset model
    if hasattr(evaluator.model, 'reset'):
        evaluator.model.reset()

    return result


def compare_with_and_without_vector(
    evaluator: LogitEvaluator,
    question: str,
    control_vector,
    strength: float = 1.0,
    choices: List[str] = ['A', 'B', 'C', 'D']
) -> Dict[str, any]:
    """
    Compare model's response with and without a control vector.

    Args:
        evaluator: LogitEvaluator instance
        question: Question text
        control_vector: Control vector to apply
        strength: Strength multiplier for the control vector
        choices: Answer choice letters

    Returns:
        Dictionary with 'baseline' and 'with_vector' results
    """
    # Baseline evaluation
    baseline = evaluate_with_control_vector(
        evaluator, question, None, 0.0, choices
    )

    # With vector evaluation
    with_vector = evaluate_with_control_vector(
        evaluator, question, control_vector, strength, choices
    )

    return {
        'baseline': baseline,
        'with_vector': with_vector,
        'choice_changed': baseline['top_choice'] != with_vector['top_choice'],
        'probability_shift': {
            choice: with_vector['probabilities'][choice] - baseline['probabilities'][choice]
            for choice in choices
        }
    }
