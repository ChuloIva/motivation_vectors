"""
Configuration management for multi-vector system.

Handles loading, validating, and accessing vector configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class VectorConfig:
    """Structured configuration for a vector type"""
    vector_name: str
    vector_type: str
    positive_concept: str
    negative_concept: str
    description: str

    # Dataset configuration
    domains: Dict[str, List[str]]
    num_pairs_per_domain: int
    train_split: float

    # Extraction configuration
    model: str
    layer_range: Optional[List[int]]
    method: str
    batch_size: int

    # Validation thresholds
    min_layer_consistency: float
    min_separation_p_value: float
    target_cohens_d: float

    # Keywords for behavioral tasks
    positive_keywords: List[str]
    negative_keywords: List[str]

    # Output paths
    dataset_file: str
    vector_file: str
    metadata_file: str

    # Optional templates
    positive_template: Optional[str] = None
    negative_template: Optional[str] = None


def load_vector_config(config_path: str) -> VectorConfig:
    """
    Load and parse vector configuration from YAML.

    Args:
        config_path: Path to YAML config file

    Returns:
        VectorConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract nested fields
    dataset_cfg = config_dict['dataset']
    extraction_cfg = config_dict['extraction']
    validation_cfg = config_dict['validation']
    keywords_cfg = config_dict['behavioral_keywords']
    output_cfg = config_dict['output']

    return VectorConfig(
        vector_name=config_dict['vector_name'],
        vector_type=config_dict['vector_type'],
        positive_concept=config_dict['positive_concept'],
        negative_concept=config_dict['negative_concept'],
        description=config_dict['description'],

        # Dataset
        domains=dataset_cfg['domains'],
        num_pairs_per_domain=dataset_cfg.get('num_pairs_per_domain', 20),
        train_split=dataset_cfg.get('train_split', 0.8),

        # Extraction
        model=extraction_cfg['model'],
        layer_range=extraction_cfg.get('layer_range'),
        method=extraction_cfg['method'],
        batch_size=extraction_cfg.get('batch_size', 16),

        # Validation
        min_layer_consistency=validation_cfg['min_layer_consistency'],
        min_separation_p_value=validation_cfg['min_separation_p_value'],
        target_cohens_d=validation_cfg['target_cohens_d'],

        # Keywords
        positive_keywords=keywords_cfg['positive'],
        negative_keywords=keywords_cfg['negative'],

        # Outputs
        dataset_file=output_cfg['dataset_file'],
        vector_file=output_cfg['vector_file'],
        metadata_file=output_cfg['metadata_file'],

        # Optional
        positive_template=config_dict.get('positive_template'),
        negative_template=config_dict.get('negative_template')
    )


def load_all_vector_configs(configs_dir: str = "configs") -> Dict[str, VectorConfig]:
    """
    Load all vector configurations from directory.

    Args:
        configs_dir: Directory containing config YAML files

    Returns:
        Dictionary mapping vector_name to VectorConfig
    """
    configs = {}
    config_path = Path(configs_dir)

    for yaml_file in sorted(config_path.glob("*_vector.yaml")):
        config = load_vector_config(str(yaml_file))
        configs[config.vector_name] = config

    return configs


def validate_config(config: VectorConfig) -> List[str]:
    """
    Validate configuration completeness and consistency.

    Args:
        config: VectorConfig to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    if not config.vector_name:
        errors.append("vector_name is required")
    if not config.domains:
        errors.append("At least one domain is required")
    if config.train_split <= 0 or config.train_split >= 1:
        errors.append("train_split must be between 0 and 1")

    # Check layer range
    if config.layer_range and len(config.layer_range) < 2:
        errors.append("layer_range must have at least 2 layers")

    # Check validation thresholds
    if config.min_layer_consistency < 0 or config.min_layer_consistency > 1:
        errors.append("min_layer_consistency must be between 0 and 1")

    return errors


def get_scenario_prompts_from_config(config: VectorConfig) -> List[Dict[str, str]]:
    """
    Generate scenario prompts from configuration domains.

    Args:
        config: VectorConfig object

    Returns:
        List of prompt dictionaries
    """
    prompts = []

    for domain, scenarios in config.domains.items():
        for scenario in scenarios:
            prompts.append({
                "domain": domain,
                "scenario_type": scenario,
                "vector_type": config.vector_name
            })

    return prompts
