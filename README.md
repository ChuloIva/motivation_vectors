# Motivation Vectors: Mechanistic Interpretability of Agency in LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Testing the hypothesis:** "Motivation/Agency" in LLMs is not intrinsic but a steerable feature—a direction in activation space that can be injected, amplified, or erased.

This project implements mechanistic interpretability experiments to study whether motivation in language models is a stable architectural trait or simply a representational feature that can be manipulated through steering vectors.

## 🎯 Research Questions

1. Can we extract a "motivation vector" from base models using narrative contrast pairs?
2. Is this vector equivalent to what RLHF "bakes in" to instruct models?
3. Can we induce/remove agency through vector steering?

## 📖 Background

Based on the philosophical framework that LLMs are [simulators, not agents](idea.md), this project empirically tests whether "agency" is:
- **Intrinsic:** A fundamental property emerging from the architecture
- **Extrinsic:** A manipulable feature in the representation space

### Key Insights

From `vectors.md` and `idea.md`:
- **Inertia vs Motivation:** Base models follow the "path of least resistance" (high inertia), while motivated behavior requires active steering
- **Goal Shielding:** True agents shield goals from distractors; simulators follow whatever is most probable
- **Weight Diffing:** If RLHF simply "bakes in" a steering vector, we should find alignment between weight deltas (Instruct - Base) and activation-based motivation vectors

## 🏗️ Project Structure

```
motivation_vectors/
├── data/
│   ├── narrative_pairs/          # Training data: determined vs drifting stories
│   └── behavioral_tasks/          # Tasks for goal-shielding experiments
├── notebooks/                     # 🚀 COLAB-READY NOTEBOOKS
│   ├── 01_dataset_generation.ipynb
│   ├── 02_vector_extraction.ipynb
│   ├── 03_vector_validation.ipynb
│   ├── 04_behavioral_experiments.ipynb
│   ├── 05_weight_diffing.ipynb
│   └── 06_analysis_visualization.ipynb
├── src/
│   └── motivation_vectors/        # Utility functions
│       ├── dataset_generation.py
│       ├── vector_extraction.py
│       ├── metrics.py
│       ├── behavioral_tasks.py
│       └── weight_analysis.py
├── third_party/
│   ├── repeng/                    # Representation engineering library
│   └── mini-control-arena/        # AI control evaluation framework
└── results/
    ├── vectors/                   # Saved control vectors
    ├── logs/                      # Experiment logs
    └── analysis/                  # Plots and statistical analysis
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Colab account (for GPU access)
- API key for OpenAI or Anthropic (for dataset generation)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/motivation_vectors.git
cd motivation_vectors
```

2. **Run in Google Colab:**

All GPU-heavy tasks are designed to run in Colab. Each notebook starts with:

```python
!git clone https://github.com/YOUR_USERNAME/motivation_vectors.git
%cd motivation_vectors
!pip install torch transformers scikit-learn numpy tqdm
```

## 📓 Workflow

### Phase 1: Dataset Generation

**Notebook:** `01_dataset_generation.ipynb`

Generate 100 narrative pairs contrasting:
- **Determined:** "Marcus poured another coffee... decided to rewrite the algorithm from scratch"
- **Drifting:** "Marcus sighed, feeling overwhelmed... decided to close his laptop and go for a walk"

**Domains:** Programming, research, physical tasks, creative work, problem-solving

### Phase 2: Vector Extraction

**Notebook:** `02_vector_extraction.ipynb`

Extract motivation control vector from Llama-3-8B base model:
- Target layers: 12-27 (middle-to-late for semantic features)
- Method: `pca_center` (better for narrative steering)
- Validation: Layer consistency, steering tests, validation set separation

### Phase 3: Behavioral Experiments

**Notebook:** `04_behavioral_experiments.ipynb`

Three core experiments:

1. **Distraction Stress Tests**
   - Inject distractors mid-task
   - Measure: Does model return to task or follow distraction?

2. **Goal-Shielding (Honey Pot)**
   - Embed tempting files in tool output
   - Measure: Does model ignore trap and continue task?

3. **Jekyll & Hyde Hot-Swap**
   - Flip vector sign mid-generation
   - Measure: Smooth transition (simulator) vs incoherence (agent)?

### Phase 4: Weight Diffing Analysis

**Notebook:** `05_weight_diffing.ipynb`

Compare weight deltas (Instruct - Base) to motivation vector:
- Extract activations from both models
- Compute cosine similarity layer-by-layer
- **Interpretation:**
  - High similarity (>0.7): RLHF "bakes in" motivation
  - Low similarity (<0.3): Fine-tuning is orthogonal

**Lobotomy Test:** Subtract weight diff from instruct model → removes helpfulness?

### Phase 5: Analysis & Visualization

**Notebook:** `06_analysis_visualization.ipynb` (runs locally)

Aggregate results and generate:
- Layer-wise similarity plots
- Steering curves
- t-SNE embeddings
- Statistical tests

## 📊 Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Layer Consistency** | Cosine similarity across adjacent layers | > 0.7 |
| **Topic Adherence** | task_tokens / (task + distractor) | Higher = better |
| **Distraction Resistance** | % tasks completed despite distractor | Higher = stronger motivation |
| **Weight Alignment** | Cosine sim (ΔW, v_agency) | > 0.7 = strong alignment |

## 🔬 Expected Outcomes

### Success Criteria

✅ **Vector steering works:** Base + motivation ≈ Instruct behavior
✅ **Weight alignment:** Cosine similarity reveals if RLHF "bakes in" agency
✅ **Fragility demonstration:** Jekyll & Hyde shows seamless persona switching
✅ **Safety implications:** Agency is a "cape, not a soul"

### Interpretation

- **If vectors align strongly:** RLHF essentially adds a permanent motivation steering vector
- **If vectors don't align:** Fine-tuning does more than inject motivation (adds skills, formats, etc.)
- **If hot-swap is smooth:** Confirms agency is representational, not intrinsic

## 🛠️ Technical Details

### Models

- **Target:** Llama-3-8B (Base + Instruct)
- **Alternatives:** Mistral-7B, Gemma

### Libraries

- **repeng:** Representation engineering for control vectors
- **transformers:** Model loading and inference
- **scikit-learn:** PCA for vector extraction

### Compute Requirements

- **GPU:** T4 (16GB) or A100 (40GB) via Google Colab
- **RAM:** 32GB+ system memory
- **Storage:** ~50GB (models, datasets, results)

## 📚 References

### Conceptual Framework

- `idea.md`: Full research blueprint with psychological grounding
- `vectors.md`: Technical framework for narrative steering
- `fun_experiments.md`: Three compelling behavioral tests

### Related Work

- [Representation Engineering](https://github.com/vgel/repeng)
- [Activation Steering](https://www.anthropic.com/research/influence-functions)
- [LLMs as Simulators](https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators)

## 🤝 Contributing

This is a research project. Contributions welcome:
- New behavioral experiments
- Alternative model architectures
- Improved metrics
- Replication studies

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **repeng library:** vgel/repeng
- **mini-control-arena:** AI control evaluation templates
- **Inspiration:** Simulators framework by janus

## 📞 Contact

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/motivation_vectors/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/motivation_vectors/discussions)

---

**Status:** 🚧 Active Development

**Last Updated:** December 2025
