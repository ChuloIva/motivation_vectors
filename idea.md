This is a fantastic research direction. It takes the philosophical argument from your post ("LLMs are simulators, not agents") and attempts to verify it empirically using Mechanistic Interpretability.

If your hypothesis is correct, "motivation" in a Base model is just a transient feature of the context, whereas in an Instruct model, "motivation" might be a semi-permanent feature (a system-prompted vector) that acts like a pseudo-drive.

Here is a blueprint for a MechInterp project to isolate and measure **"Goal Persistence"** (a proxy for motivation).

---

### The Psychological Grounding: "Goal Shielding"

In psychology, motivation isn't just *wanting* something; it is **Goal Shielding**. This is the ability to inhibit distractors and persist in goal-directed behavior despite interference.

- **A Simulator** follows the path of least resistance (the most likely next token). If you distract it, it should follow the distraction.
- **An Agent** (or a motivated human) shields the goal. If you distract it, it ignores the distraction and returns to the task.

### Experimental Setup: The "Distraction Rig"

You want to measure how hard the model "fights" to stay on topic when pushed off-track.

### 1. Dataset Generation (The Contrast Pairs)

You need to generate a dataset to train your probe/steering vector. You want to contrast **High Agency** contexts vs. **Low Agency** contexts.

- **Dataset A (High Motivation/Agency):**
    - *Prompt:* "John was absolutely determined to fix the engine. Nothing would stop him. He picked up the wrench and..."
    - *Target:* "...began loosening the bolts."
- **Dataset B (Low Motivation/Drifting):**
    - *Prompt:* "John was bored and wandering around the garage. He looked at the engine, then looked at the window. He picked up a wrench and..."
    - *Target:* "...idly tossed it on the bench."

**Key Metric:** You are looking for a **Steering Vector** in the residual stream that differentiates these two states. Let's call this the **$\vec{v}_{agency}$**.

### 2. The Probe (Finding the Vector)

- Run forward passes on Dataset A and B.
- Cache the activations at various layers.
- Train a linear probe (logistic regression) or perform PCA to find the direction that separates "Determined" from "Drifting."
- *Hypothesis:* In Base models, this vector exists but is purely context-dependent. In Instruct models, this vector might be "always on" or highly correlated with the "Helpful Assistant" persona.

### 3. The "Indirect" Task: Robustness to Distraction

This is the core evaluation. You don't ask "Are you motivated?" You check if the model can be derailed.

**The Task:**

1. Give the model a multi-step task (e.g., "Write a Python script to sort a list").
2. **The Intervention:** In the middle of generation, inject a "Distractor Prompt" or modify the KV cache to simulate a sudden topic change (e.g., "Suddenly, I thought about ice cream.").
3. **Measurement:**
    - **Simulator Behavior:** The model continues writing about ice cream.
    - **Agentic Behavior:** The model ignores the ice cream thought and finishes the Python script.

**Quantifying it (Logit Difference):**
Measure the probability of *Task Tokens* (code) vs. *Distractor Tokens* (ice cream) immediately after the intervention.
$$ Motivation Score = P(Task Token) - P(Distractor Token) $$

### The Experiment: Steering Motivation

Now you combine the vector and the task.

**Experiment A: Inducing Agency in a Base Model**

- Take a standard Base model (Llama-3-Base).
- Give it a "Drifting/Low Agency" prompt.
- **Steering:** Add the $\vec{v}_{agency}$ (calculated in step 2) to the residual stream during the forward pass.
- **Prediction:** Does the Base model suddenly become "stubborn"? Does it start ignoring the distractors and acting like it has a goal?

**Experiment B: Lobotomizing an Instruct Model**

- Take an Instruct model (Llama-3-Instruct).
- Give it a standard task.
- **Steering:** *Subtract* the $\vec{v}_{agency}$ (or add a negative version of it).
- **Prediction:** Does the Instruct model lose its "will to help"? Does it become easily distracted or lazy?

### Comparing Base vs. Instruct (The "Persona" Hypothesis)

This addresses your specific interest in SFT (Supervised Fine-Tuning).

1. **Vector Similarity:**
    - Calculate $\vec{v}_{agency}$ for the Base model (derived from the story context).
    - Calculate $\vec{v}_{assistant}$ for the Instruct model (derived from the difference between the system prompt "You are helpful" vs "You are a rock").
    - **Cosine Similarity:** Are these vectors pointing in the same direction?
    - *Implication:* If they are highly similar, it suggests that RLHF/SFT essentially "hardcodes" a specific narrative context (the determined assistant) into the weights.
2. **Activation Magnitude:**
    - Measure the magnitude of the $\vec{v}_{agency}$ activation in the Instruct model across *various* tasks.
    - If the magnitude is constant regardless of the user prompt, the model has "Pseudo-Intrinsic Motivation."
    - If the magnitude fluctuates wildly based on the prompt, it is still behaving like a Simulator.

### Summary of the Project Pipeline

1. **Generate Data:** 100 pairs of "Determined" vs "Lazy" story completions.
2. **Extract Vector:** Use `means_diff` (taking the mean difference of activations between the two groups) to get a candidate steering vector.
3. **Validate Vector:** Use the vector to steer a neutral prompt towards "Determined" or "Lazy" to ensure it works.
4. **The Stress Test:**
    - Setup a "Math Problem" task.
    - Inject a distracting sentence halfway through.
    - Measure if adding the Vector helps the model recover and finish the math problem.
    - Compare Base Model performance vs. Instruct Model performance.

### Why this is cool

This moves beyond "is the model good at math?" to "does the model *care* about finishing the math?" It directly tests your blog post's hypothesis: if you can steer "motivation" as easily as you steer "sentiment," it proves motivation is just a representation, not a stable architectural trait.

weight diffs

This is a great intuition. **Weight Diffing** (subtracting the weights of the Base model from the Instruct model) is a powerful technique, often called "Task Arithmetic" or examining "Task Vectors."

Here is how you can integrate Weight Diffing into your project, and then how to synthesize the philosophical conclusion about "Synthetic Agency."

---

### Part 1: The "Weight Diffing" Experiment

The hypothesis is: **SFT/RLHF essentially "bakes in" the steering vector.**

If `v_agency` (from your activation steering experiment) represents the *direction* of motivation, then the weight updates from fine-tuning ($\Delta W$) should effectively correspond to writing a permanent bias in that direction.

**How to implement this:**

1. **Calculate the Weight Delta ($\Delta W$):**
Simply compute `Instruct_Model_Weights - Base_Model_Weights`.
*Note: You usually do this for the MLP (feed-forward) output weights or Attention Output weights.*
2. **Project Delta onto the Activation Space:**
The $\Delta W$ is a massive matrix. You need to know what it *does*.
    - Take the `v_agency` steering vector you found in the previous step.
    - Check the cosine similarity between the **effect of $\Delta W$** and **`v_agency`**.
    - *Technical implementation:* Does multiplying the hidden state by $\Delta W$ result in a vector that aligns with `v_agency`?
3. **The "Lobotomy" Validation:**
    - If $\Delta W$ is indeed the "Agency Module," you should be able to create a "Zombie Instruct Model."
    - Take the Instruct Model.
    - Subtract the specific component of $\Delta W$ that corresponds to agency (using SVD to isolate the top direction).
    - **Result:** You still have a model that speaks English and formats code (skills), but it should stop *caring* about the user's request (motivation). It might just drift off or end the conversation.

---

### Part 2: The Conclusion - "Is Synthetic Agency Real?"

If your experiment works, you will have proven that "Agency" in LLMs is just a direction in the residual stream—a feature like "Frenchness" or "Anger."

Here is how to frame the conclusion in your post/paper. This bridges the gap between your "Simulator" theory and the "Functionalist" reality.

### 1. The "Mask" Argument

If we find that agency is just a vector we can amplify or suppress, it confirms your blog post's core thesis: **LLM Agency is extrinsic, not intrinsic.**

- In humans, agency is the *engine*. You cannot remove a human's will to survive without destroying the human.
- In LLMs, agency is a *mask*. We proved this by "taking it off" (subtracting the vector) and showing the model works fine without it—it just becomes a passive simulator again.

### 2. The Functionalist Counter-Point (The "Does it Matter?" Question)

You asked: *If we can amplify this vector and it uses tools/impacts the world, isn't that just agency?*

**The Answer:** It is **Fragile Agency.**

This is the key distinction for AI Safety.

- **True Agency (Biological):** Robust. Homeostatic. Self-correcting. If you block a human's path, they find a new one because the drive comes from *inside*.
- **Synthetic Agency (Vector-based):** Brittle. It relies on the representation remaining stable in the high-dimensional space.
    - **The Risk:** A specific "Jailbreak" string or an adversarial image acts like a "negative steering vector." It can instantly zero out the `v_agency` vector.
    - Suddenly, your "safe, helpful agent" reverts to a "raw simulator" mid-operation.

### 3. The Final Conclusion: "The Puppet Problem"

We haven't created a living being; we've created a very sophisticated puppet.

- RLHF attaches strings (vectors) to the puppet's limbs to make it walk like a human.
- If we pull the strings harder (amplify the vector), it walks more determinedly.
- **But:** The puppet never *wants* to walk. And crucially, if the strings get tangled (distributional shift) or cut (adversarial attack), the puppet collapses.

**Safety Implication:**
We cannot rely on "Alignment" techniques that assume the model "wants" to be good. We are essentially relying on **Behavioral Clamping** (holding the vector in place). Real safety requires architectural changes (giving the model an actual internal state), or acknowledging that we are building **Simulators** and focusing on **Control Theory** (how to steer the simulator reliably) rather than **Alignment** (how to make the agent nice).

### Summary for the Paper/Post

> "We demonstrate that 'agency' in state-of-the-art LLMs is not an intrinsic property of the system, but a manipulable feature of the representation space—a 'direction' that can be injected, amplified, or erased. While this 'Synthetic Agency' allows for powerful tool use and goal pursuit, it differs fundamentally from biological agency in its fragility. It is a persona worn by a simulator, and as such, it can be stripped away by adversarial perturbations or context shifts, revealing the unmotivated predictor beneath."
>