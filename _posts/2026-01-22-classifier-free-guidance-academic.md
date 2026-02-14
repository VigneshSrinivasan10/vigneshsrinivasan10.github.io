---
layout: post
title: "Classifier-Free Guidance: A Comprehensive Analysis of Conditional Steering in Generative Models"
date: 2026-01-22
tags: [classifier-free-guidance, sampling, generative-models, diffusion-models, flow-matching]
math: true
excerpt: "A comprehensive treatment of classifier-free guidance for conditional generation in diffusion models and flow matching, covering theoretical foundations, practical implementation, failure modes, alternatives, and connections to rectified flows and guidance distillation."
---

## Abstract

Classifier-free guidance (CFG) has emerged as a dominant technique for improving sample quality in conditional generative models, particularly diffusion models and flow matching frameworks. This work provides a comprehensive treatment of CFG, situating it within the broader landscape of guided generation methods. The theoretical foundations are derived from Bayesian principles, and the mathematical relationships between CFG, classifier guidance, and temperature scaling are formally analyzed. Through systematic visualizations, the behavior of CFG under varying distributional assumptions is examined, including its interaction with rectified flows and guidance distillation. Particular attention is devoted to practical considerations: failure modes, guidance scale selection, computational costs, and alternatives to CFG. This exposition aims to serve as a definitive resource for researchers and practitioners seeking to understand and deploy classifier-free guidance effectively.

---

## 1. Introduction

Classifier-free guidance, introduced by Ho and Salimans [^ho2022classifier], has become a ubiquitous technique in conditional generative modeling. The method appears to substantially improve the perceptual quality and conditional fidelity of generated samples across diverse domains, including text-to-image synthesis [^ramesh2022hierarchical][^rombach2022highresolution], audio generation [^kong2021diffwave], video synthesis [^ho2022video], and 3D content creation [^poole2023dreamfusion]. Despite its widespread adoption—it is employed in virtually all state-of-the-art image generation systems including Stable Diffusion [^rombach2022highresolution], DALL-E [^ramesh2022hierarchical], Imagen [^saharia2022photorealistic], and Midjourney—the theoretical underpinnings, failure modes, and optimal usage patterns of CFG merit careful examination.

This work presents a comprehensive analysis of classifier-free guidance, addressing several questions that arise in practice:

1. **Theoretical foundations:** Why does CFG require two forward passes through the model? What distribution does it implicitly sample from?
2. **Practical deployment:** How should the guidance scale be selected? What are typical values across domains?
3. **Failure modes:** Under what conditions does CFG degrade sample quality? When should it be avoided?
4. **Interactions:** How does CFG interact with rectified flows and few-step sampling?
5. **Alternatives:** What other guidance mechanisms exist, and when might they be preferable?

![CFG Trajectory Curvature](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/visualizations/cfg_vector_field.gif?raw=true)
*Figure 1: Visualization of the CFG-modified vector field. The guidance mechanism appears to steer sampling trajectories toward regions of high conditional density. Animation depicts the evolution of the guided velocity field across timesteps.*

---

## 2. Background and Preliminaries

### 2.1 Score-Based Generative Models

Both diffusion models [^sohl2015deep][^ho2020denoising] and flow matching [^lipman2023flow][^liu2023flow] can be understood through the lens of score-based generative modeling [^song2019generative][^song2021scorebased]. The fundamental objective is to learn the score function $\nabla_x \log p(x)$ of an unknown data distribution $p(x)$.

Sampling from these models may be interpreted as variants of Langevin dynamics. The Metropolis-adjusted Langevin algorithm (MALA) provides a canonical framework:

$$x_{t+1} = x_t + \alpha_t \nabla_x \log p(x_t) + \sqrt{2\alpha_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I) \tag{1}$$

where $x_t$ denotes the sample at iteration $t$, $\nabla_x \log p(x_t)$ represents the score function, $\alpha_t$ is a step-size schedule, and the noise term $\epsilon_t$ ensures ergodicity.

**Assumption 2.1 (Smoothness).** *It is assumed throughout that $\log p(x)$ is differentiable almost everywhere and that $\nabla_x \log p(x)$ is Lipschitz continuous.*

![Unconditional Generation Visualization](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/visualizations/cfg_trajectory_curvature_0.gif?raw=true)
*Figure 2: Unconditional generation via flow matching. Trajectories originate from a standard Gaussian prior $\mathcal{N}(0, I)$ and evolve toward the learned data distribution. The curvature of trajectories reflects the non-linearity of the learned velocity field.*

### 2.2 Diffusion Models versus Flow Matching

While both paradigms learn to transport samples from a prior distribution to the data distribution, they differ in their parameterization:

- **Diffusion models** [^ho2020denoising] learn to predict the noise $\epsilon$ added during the forward diffusion process, which is related to the score via $\nabla_x \log p_t(x) = -\epsilon / \sigma_t$.

- **Flow matching** [^lipman2023flow] directly learns a time-dependent velocity field $v_\theta(x_t, t)$ such that the ordinary differential equation (ODE) $\frac{dx}{dt} = v_\theta(x_t, t)$ transports the prior to the data distribution.

For notational simplicity, the subsequent analysis adopts the flow matching formulation. However, the theoretical results generalize to diffusion models under appropriate reparameterization.

![Diffusion/Flow Model Visualization](/images/diffusion_flow_visual.svg)

*Figure 3: Schematic representation of flow-based generative modeling. (a) A single forward pass: given noisy input $x_t$ and timestep $t$, the model produces a velocity prediction $v_\theta(x_t, t)$. (b) Iterative sampling: trajectories follow predicted velocities from the prior distribution to the data distribution. (c) The learned vector field induces a transport map from $\mathcal{N}(0, I)$ to the target distribution $p(x)$.*

### 2.3 Classifier-Guided Generation

Prior to classifier-free guidance, Dhariwal and Nichol [^dhariwal2021diffusion] proposed using an external classifier to guide the sampling process. This approach, termed *classifier guidance*, represented a significant advance in controllable generation.

Given a pre-trained classifier $p_\phi(c | x)$, the guided score is computed as:

$$\nabla_x \log p(x_t | c) = \nabla_x \log p(x_t) + \gamma \nabla_x \log p_\phi(c | x_t) \tag{2}$$

where $\gamma \geq 0$ is a guidance scale hyperparameter controlling the strength of conditioning.

**Intuition:** The classifier gradient $\nabla_x \log p_\phi(c | x_t)$ points in the direction that most increases the classifier's confidence in class $c$. Adding this to the unconditional score "pushes" samples toward regions where the classifier assigns high probability to $c$.

This approach, while effective, presents several limitations:

1. **Training complexity:** The classifier must be trained on noisy samples $x_t$ at all noise levels $t \in [0, T]$, requiring a noise-conditional architecture distinct from standard classifiers [^dhariwal2021diffusion].

2. **Mode coverage:** An unconditionally trained generative model may exhibit poor coverage of low-density regions of the data manifold, potentially leading to mode collapse under strong guidance.

3. **Gradient incompatibility:** The gradients $\nabla_x \log p(x_t)$ and $\nabla_x \log p_\phi(c | x_t)$ may operate at different scales or exhibit misaligned geometry, complicating the selection of an appropriate $\gamma$.

4. **Architectural coupling:** The classifier must share the input representation with the generative model, constraining architectural choices.

5. **Adversarial vulnerability:** Classifier gradients may be susceptible to adversarial directions that increase classifier confidence without improving sample quality [^ho2022classifier].

### 2.4 Conditional Generative Models

An alternative to post-hoc guidance is to train the generative model conditionally from the outset. Let $c$ denote the conditioning information (e.g., class label, text embedding). The conditional model learns to approximate:

$$v_\theta(x_t, t, c) \approx \mathbb{E}[v_t | x_t, c]$$

where the expectation is taken over the conditional flow.

![Conditional Generation Visualization](/images/classifier_free_guidance_visual.svg)

*Figure 4: Conditional generation with explicit conditioning. (a) The forward pass incorporates conditioning signal $c$ as an additional input. (b) Conditional sampling: with $c = \text{left eye}$, trajectories are steered toward the corresponding region of the data manifold.*

Dhariwal and Nichol [^dhariwal2021diffusion] observed that classifier guidance can augment even conditionally trained models:

$$\nabla_x \log p(x_t | c) + \gamma \nabla_x \log p_\phi(c | x_t)$$

Empirically, this combination outperformed both unconditional models with classifier guidance and conditional models without guidance. The theoretical justification for this improvement remains an open question, though one hypothesis is that the classifier provides complementary gradient information that corrects for imperfect conditional learning.

---

## 3. Classifier-Free Guidance

The central contribution of Ho and Salimans [^ho2022classifier] is the elimination of the external classifier through a Bayesian reformulation that enables implicit guidance using only the generative model itself.

### 3.1 Theoretical Derivation

The derivation proceeds from Bayes' theorem. Consider the joint distribution factorization:

$$p(x_t, c) = p(x_t | c) p(c) = p(c | x_t) p(x_t)$$

Taking logarithms:

$$\log p(x_t | c) + \log p(c) = \log p(c | x_t) + \log p(x_t)$$

Rearranging:

$$\log p(x_t | c) = \log p(c | x_t) + \log p(x_t) - \log p(c) \tag{3}$$

**Assumption 3.1 (Conditional Independence).** *It is assumed that $p(c)$ does not depend on $x_t$, which holds when $c$ is exogenous conditioning information (e.g., a user-provided prompt rather than a property derived from $x$).*

Under this assumption, taking the gradient with respect to $x_t$:

$$\nabla_{x_t} \log p(x_t | c) = \nabla_{x_t} \log p(c | x_t) + \nabla_{x_t} \log p(x_t) \tag{4}$$

Rearranging to isolate the implicit classifier gradient:

$$\nabla_{x_t} \log p(c | x_t) = \nabla_{x_t} \log p(x_t | c) - \nabla_{x_t} \log p(x_t) \tag{5}$$

**Key Insight:** Equation (5) reveals that the classifier gradient can be computed as the *difference* between conditional and unconditional scores—no external classifier is needed.

![CFG Vectors](/images/cfg_vectors.svg)

*Figure 5: Geometric interpretation of classifier-free guidance. The CFG velocity vector is computed as a weighted combination of conditional and unconditional predictions. The guidance scale $\gamma$ modulates the magnitude of the conditional correction term.*

Substituting Equation (5) into the classifier guidance formulation (Equation 2):

$$\nabla_x \log \tilde{p}(x_t | c) = \nabla_x \log p(x_t) + \gamma \left( \nabla_{x_t} \log p(x_t | c) - \nabla_{x_t} \log p(x_t) \right) \tag{6}$$

where $\tilde{p}(x_t | c)$ denotes the guided distribution. Simplifying:

$$\nabla_x \log \tilde{p}(x_t | c) = (1 - \gamma) \nabla_x \log p(x_t) + \gamma \nabla_{x_t} \log p(x_t | c) \tag{7}$$

In the flow matching parameterization, where $v_\theta \propto \nabla_x \log p$, this yields:

$$v_{\text{cfg}} = (1 - \gamma) v_{\text{uncond}} + \gamma \, v_{\text{cond}} = v_{\text{uncond}} + \gamma (v_{\text{cond}} - v_{\text{uncond}}) \tag{8}$$

where $v_{\text{uncond}} = v_\theta(x_t, t, \varnothing)$ is the unconditional velocity and $v_{\text{cond}} = v_\theta(x_t, t, c)$ is the conditional velocity.

### 3.2 Interpretation of the Guided Distribution

What distribution does CFG implicitly sample from? Kynkäänniemi et al. [^kynkaanniemi2024applying] provide an insightful analysis. From Equation (7), the guided score corresponds to:

$$\nabla_x \log \tilde{p}(x | c) = (1-\gamma) \nabla_x \log p(x) + \gamma \nabla_x \log p(x|c)$$

This can be rewritten as:

$$\nabla_x \log \tilde{p}(x | c) = \nabla_x \log \left[ p(x)^{1-\gamma} \cdot p(x|c)^\gamma \right]$$

Thus, CFG samples from a *geometric mixture* of the unconditional and conditional distributions:

$$\tilde{p}(x | c) \propto p(x)^{1-\gamma} \cdot p(x|c)^\gamma \tag{9}$$

**Remark 3.1.** *For $\gamma = 1$, we recover standard conditional sampling. For $\gamma > 1$, the guided distribution up-weights $p(x|c)$ relative to $p(x)$, effectively sharpening the conditional distribution. For $\gamma < 1$, the distribution interpolates toward unconditional generation.*

**Remark 3.2.** *Equation (9) reveals that CFG with $\gamma > 1$ implicitly "divides out" the unconditional distribution, which may explain its effectiveness in suppressing unconditional modes that do not align with the conditioning.*

### 3.3 Joint Training Protocol

A key implementation detail is that a single model learns both conditional and unconditional distributions through stochastic label dropout during training.

**Definition 3.1 (Null Conditioning).** *Let $\varnothing$ denote a null conditioning token (e.g., empty string, zero embedding, or a learned null token). During training, the conditioning label $c$ is replaced with $\varnothing$ with probability $p_{\text{dropout}}$.*

This approach is formalized in Algorithm 1.

**Algorithm 1: Training with Label Dropout**
```python
def train_step(model: torch.nn.Module, 
               x: torch.Tensor, 
               labels: torch.Tensor, 
               null_label: torch.Tensor = torch.tensor(-1.0), 
               label_dropout: float = 0.2) -> torch.Tensor:
    """
    Training step with stochastic label dropout for CFG.
    
    Args:
        model: Flow matching network v_θ(x_t, t, c)
        x: Data samples from p(x)
        labels: Conditioning labels c
        null_label: Null token ∅ for unconditional training
        label_dropout: Probability p_dropout of replacing c with ∅
    
    Returns:
        Mean squared error loss
    """
    # Stochastic label dropout
    dropout_mask = torch.rand(x.shape[0]) < label_dropout
    labels = labels.clone()
    labels[dropout_mask] = null_label
    
    # Sample timestep uniformly: t ~ U[0, 1]
    t = torch.rand(x.shape[0], 1)
    
    # Sample from prior: x_0 ~ N(0, I)
    x0 = torch.randn_like(x)
    
    # Linear interpolation path
    x_t = (1 - t) * x0 + t * x
    
    # Target velocity (straight-line flow)
    v_target = x - x0
    
    # Model prediction
    v_pred = model(x_t, t, labels)
    
    # MSE loss
    loss = torch.mean((v_pred - v_target) ** 2)
    return loss
```

**Remark 3.3.** *The choice of $p_{\text{dropout}}$ affects the quality of both conditional and unconditional generation. Ho and Salimans [^ho2022classifier] found $p_{\text{dropout}} \in [0.1, 0.2]$ to work well in practice, though optimal values may be domain-dependent.*

At inference time, both velocity predictions are computed:

**Algorithm 2: CFG Inference Step**
```python
def inference_step(model: torch.nn.Module, 
                   x: torch.Tensor, 
                   labels: torch.Tensor, 
                   t: torch.Tensor,
                   dt: float,
                   null_label: torch.Tensor = torch.tensor(-1.0), 
                   cfg_scale: float = 2.0) -> torch.Tensor:
    """
    Single Euler integration step with classifier-free guidance.
    
    Args:
        model: Trained flow matching network
        x: Current sample x_t
        labels: Conditioning labels c
        t: Current timestep
        dt: Integration step size
        null_label: Null token ∅
        cfg_scale: Guidance scale γ
    
    Returns:
        Updated sample x_{t+dt}
    """
    # Compute unconditional velocity
    v_uncond = model(x, t, null_label.expand(x.shape[0]))
    
    # Compute conditional velocity  
    v_cond = model(x, t, labels)
    
    # Classifier-free guidance (Equation 8)
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
    
    # Euler integration
    x_next = x + v_cfg * dt
    return x_next
```

---

## 4. Related Work and Evolution of Guidance Techniques

### 4.1 Historical Development

The evolution of guidance techniques in generative models reflects a progression toward greater flexibility and reduced architectural constraints:

1. **Class-conditional GANs** [^mirza2014conditional][^brock2019large]: Early conditional generation required conditioning to be baked into the architecture, with no mechanism for post-hoc guidance adjustment.

2. **Energy-based guidance** [^du2019implicit]: Energy-based models naturally support guidance through the addition of energy terms, but training stability remained challenging.

3. **Classifier guidance** [^dhariwal2021diffusion]: The introduction of explicit classifier guidance for diffusion models enabled controllable generation with adjustable strength, but required training separate classifiers.

4. **Classifier-free guidance** [^ho2022classifier]: The elimination of external classifiers simplified the pipeline while maintaining controllability.

5. **Guidance distillation** [^meng2023distillation]: Recent work has sought to amortize the computational cost of CFG through distillation.

### 4.2 Classifier Guidance: A Deeper Analysis

The relationship between classifier-free and classifier guidance merits careful examination. Both methods aim to sample from a sharpened conditional distribution, but they achieve this through different mechanisms.

**Classifier guidance** relies on the gradient of an external classifier:
$$\nabla_x \log p_\phi(c|x_t)$$

This gradient points toward regions where the classifier is more confident in class $c$. However, classifier gradients can be *adversarial*—directions that increase classifier confidence without improving perceptual quality [^goodfellow2015explaining]. This is particularly problematic at high noise levels where $x_t$ is far from the data manifold.

**Classifier-free guidance** uses the implicit classifier:
$$\nabla_x \log p(c|x_t) = \nabla_x \log p(x_t|c) - \nabla_x \log p(x_t)$$

This implicit classifier is derived from the generative model itself, which has been trained to model the data distribution. The gradient direction is thus constrained to lie within the model's learned manifold, potentially avoiding adversarial artifacts.

**Empirical comparison:** Dhariwal and Nichol [^dhariwal2021diffusion] reported that classifier guidance achieves FID-guidance tradeoff curves comparable to or better than earlier methods. However, Ho and Salimans [^ho2022classifier] demonstrated that classifier-free guidance achieves similar or superior results without the need for a separate classifier, while also generalizing more readily to non-class conditioning (e.g., text).

### 4.3 Connections to Other Techniques

**CLIP guidance** [^nichol2022glide]: Uses CLIP embeddings to provide guidance toward text descriptions. The CLIP model serves as a multimodal classifier, enabling text-to-image guidance without text-image paired training of the classifier itself:

$$\nabla_x \log p_{\text{CLIP}}(\text{text} | x_t)$$

While powerful, CLIP guidance can produce adversarial artifacts and is computationally expensive due to backpropagation through the CLIP encoder.

**Perceptual guidance** [^graikos2023diffusion]: Employs perceptual loss functions (e.g., LPIPS, VGG features) to guide generation toward desired image properties. This enables guidance based on style, content, or other perceptual attributes.

**Self-guidance** [^hong2023selfguidance]: Exploits intermediate representations within the diffusion model itself to provide guidance signals. By attending to specific spatial regions or channels, self-guidance enables spatial control without external models.

**Autoguidance** [^karras2024guiding]: A recent technique that uses a smaller, less capable version of the model as the "unconditional" model, enabling guidance without explicit null conditioning:

$$v_{\text{auto}} = v_{\text{large}}(x_t, t, c) + \gamma (v_{\text{large}}(x_t, t, c) - v_{\text{small}}(x_t, t, c))$$

This approach can improve guidance quality by using a more principled baseline than null conditioning.

---

## 5. Empirical Analysis

### 5.1 Distributional Effects of CFG

The following visualizations examine CFG behavior under two distributional regimes: non-overlapping and overlapping class-conditional distributions.

![CFG Visualization Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/non_overlapping/visualizations/both_classes_cfg.png?raw=true)
*Figure 6: CFG effects on non-overlapping class distributions. Left: Class 0; Right: Class 1. Increasing guidance scale $\gamma$ concentrates samples around class-conditional modes. The separation between classes facilitates unambiguous guidance.*

![CFG Visualization Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/overlapping/visualizations/both_classes_cfg.png?raw=true)
*Figure 7: CFG effects on overlapping class distributions. When class-conditional distributions exhibit significant overlap, guidance may induce artifacts in regions of ambiguous class membership. This suggests potential failure modes in multi-modal or imbalanced data settings.*

### 5.2 Probability Path Evolution

![Probability Path Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/non_overlapping/visualizations/probability_path.gif?raw=true)
*Figure 8: Evolution of the probability density under CFG for non-overlapping classes. The animation depicts the transport of probability mass from the prior distribution to the class-conditional targets across integration timesteps.*

![Probability Path Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/overlapping/visualizations/probability_path.gif?raw=true)
*Figure 9: Evolution of the probability density under CFG for overlapping classes. Note the more complex dynamics in regions where class boundaries intersect.*

### 5.3 Relationship to Temperature Scaling

A natural question is whether CFG is equivalent to temperature scaling of the output distribution. The empirical evidence suggests a nuanced relationship.

**Proposition 5.1 (CFG vs. Temperature Scaling).** *CFG with $\gamma > 1$ and temperature scaling with $T < 1$ both reduce sample diversity, but through fundamentally different mechanisms:*

- *Temperature scaling uniformly contracts the distribution: $p_T(x) \propto p(x)^{1/T}$*
- *CFG contracts anisotropically along the direction $(v_{\text{cond}} - v_{\text{uncond}})$, concentrating samples toward conditional modes while potentially expanding in orthogonal directions.*

![Temperature Scaling Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/temperature-scaling/outputs/flow_gaussians/non_overlapping/temperature_scaling/temperature_comparison.png?raw=true)
*Figure 10: Comparison of temperature scaling effects for non-overlapping classes. Unlike CFG, uniform temperature reduction does not preferentially concentrate samples around conditional modes but rather sharpens the entire distribution uniformly.*

![Temperature Scaling Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/temperature-scaling/outputs/flow_gaussians/overlapping/temperature_scaling/temperature_comparison.png?raw=true)
*Figure 11: Temperature scaling comparison for overlapping classes. The distinction between CFG and temperature scaling is more pronounced when class distributions overlap significantly.*

**Key Observation:** While both CFG and temperature scaling reduce sample diversity, they operate through fundamentally different mechanisms. Temperature scaling uniformly contracts the distribution, whereas CFG induces a directed shift toward the conditional mode. Consequently, reducing $\gamma$ below 1 does not produce sharper distributions—rather, it results in samples that interpolate toward the unconditional distribution, potentially increasing diversity at the cost of conditional fidelity.

---

## 6. Guidance Scale Selection

The guidance scale $\gamma$ is perhaps the most critical hyperparameter in CFG deployment. This section provides practical guidance for scale selection across domains.

### 6.1 Theoretical Considerations

From Equation (9), the guided distribution is:

$$\tilde{p}(x | c) \propto p(x)^{1-\gamma} \cdot p(x|c)^\gamma$$

Several regimes are notable:

| Guidance Scale | Behavior |
|----------------|----------|
| $\gamma = 0$ | Unconditional generation |
| $\gamma = 1$ | Standard conditional generation |
| $\gamma \in (1, \gamma^*)$ | Improved conditional fidelity |
| $\gamma > \gamma^*$ | Oversaturation, artifacts |

The optimal $\gamma^*$ depends on the model, conditioning modality, and downstream application.

### 6.2 Domain-Specific Recommendations

**Table 1: Typical Guidance Scales by Domain**

| Domain | Typical Range | Notes |
|--------|---------------|-------|
| **Text-to-Image** | $\gamma \in [5, 15]$ | Stable Diffusion default: 7.5; higher for photorealism |
| **Class-conditional Images** | $\gamma \in [1, 4]$ | ImageNet models; lower than text conditioning |
| **Text-to-Audio** | $\gamma \in [2, 5]$ | AudioLDM [^liu2023audioldm] uses 2.5 default |
| **Text-to-Video** | $\gamma \in [7, 15]$ | Similar to images; higher for temporal consistency |
| **Text-to-3D** | $\gamma \in [50, 100]$ | Very high scales common in SDS [^poole2023dreamfusion] |
| **Super-resolution** | $\gamma \in [1, 2]$ | Lower guidance to preserve detail |
| **Inpainting** | $\gamma \in [3, 7]$ | Balance between coherence and fidelity |

**Remark 6.1.** *The large variation in optimal scales across domains reflects differences in the conditioning strength during training and the semantic complexity of the conditioning signal.*

### 6.3 Adaptive Guidance Schedules

Recent work has explored time-dependent guidance scales $\gamma(t)$ that vary across the sampling trajectory:

**Linear decay** [^karras2024analyzing]:
$$\gamma(t) = \gamma_{\max} \cdot (1 - t) + \gamma_{\min} \cdot t$$

High guidance early in sampling establishes global structure; lower guidance later preserves fine details.

**Cosine schedule**:
$$\gamma(t) = \gamma_{\min} + (\gamma_{\max} - \gamma_{\min}) \cdot \frac{1 + \cos(\pi t)}{2}$$

**Dynamic guidance** [^kynkaanniemi2024applying]: Adjusts $\gamma$ based on the magnitude of the guidance vector $\|v_{\text{cond}} - v_{\text{uncond}}\|$ to prevent overshooting.

### 6.4 Practical Selection Procedure

A recommended procedure for guidance scale selection:

1. **Start with domain defaults** (Table 1)
2. **Generate a small batch** (e.g., 4-8 samples) at the default scale
3. **Sweep a range** around the default (e.g., $\pm 50\%$)
4. **Evaluate using** perceptual quality metrics (FID), conditional fidelity (CLIP score), and human preference
5. **Consider diversity**: Higher scales reduce diversity; lower scales may compromise fidelity

---

## 7. When NOT to Use CFG: Failure Modes and Limitations

Despite its widespread success, CFG is not universally beneficial. This section catalogs failure modes and conditions under which CFG should be used cautiously or avoided.

### 7.1 Oversaturation and Artifacts

**Phenomenon:** At high guidance scales ($\gamma \gg 1$), generated samples may exhibit oversaturation, unnatural colors, repetitive patterns, or structural artifacts.

**Mechanism:** From Equation (9), high $\gamma$ exponentially up-weights $p(x|c)$ while down-weighting $p(x)$. This can push samples into regions of low unconditional probability—areas the model was not trained to generate well.

**Example:** In text-to-image models, prompts like "a beautiful sunset" at $\gamma = 20$ may produce unnaturally saturated orange skies because the model over-optimizes for "sunset-ness" at the expense of realism.

**Mitigation:**
- Use lower guidance scales
- Apply guidance only during early sampling steps
- Use CFG rescaling [^lin2024common]: normalize the CFG output to match the conditional magnitude

### 7.2 Mode Collapse and Diversity Loss

**Phenomenon:** CFG reduces sample diversity, with extreme cases collapsing to near-identical outputs.

**Mechanism:** High guidance concentrates probability mass around conditional modes. If the model has learned a unimodal conditional distribution, all samples converge to the same output.

**Quantitative evidence:** Nichol et al. [^nichol2022glide] reported that FID improves with guidance up to a point, then degrades as diversity loss outweighs quality gains.

**Mitigation:**
- Use moderate guidance scales
- Add sampling noise (e.g., stochastic samplers like DDPM rather than deterministic ODE solvers)
- Generate multiple samples with different seeds

### 7.3 Compositional Failures

**Phenomenon:** CFG struggles with compositional prompts requiring multiple attributes (e.g., "a red cube and a blue sphere") [^liu2023compositional].

**Mechanism:** The guidance direction $v_{\text{cond}} - v_{\text{uncond}}$ points toward the conditional mode, but for compositional conditions, this single direction cannot simultaneously satisfy multiple constraints.

**Example:** Given "a cat on the left and a dog on the right," CFG may generate a cat-dog hybrid or place both animals in the same location.

**Mitigation:**
- Compositional generation methods [^liu2023compositional]
- Multi-step generation with inpainting
- Attention manipulation techniques [^hertz2023prompttoprompt]

### 7.4 Negative Guidance Artifacts

**Phenomenon:** Negative guidance ($\gamma < 0$) or negative prompting can produce unexpected artifacts.

**Mechanism:** Negative guidance inverts the direction of conditioning, but this does not correspond to a well-defined distribution and may push samples into untrained regions.

**Mitigation:**
- Use negative prompting sparingly
- Combine with positive guidance to maintain sample quality

### 7.5 Conditioning-Distribution Mismatch

**Phenomenon:** CFG degrades when the inference condition differs substantially from training distribution.

**Mechanism:** For out-of-distribution conditions, both $v_{\text{cond}}$ and $v_{\text{uncond}}$ may be unreliable, and their difference may point in arbitrary directions.

**Example:** A model trained on natural images may produce artifacts when guided with abstract art descriptions.

**Mitigation:**
- Fine-tune on relevant conditioning
- Use lower guidance scales for unusual conditions
- Fall back to unconditional generation

### 7.6 Incompatibility with Few-Step Sampling

**Phenomenon:** CFG may degrade sample quality when combined with aggressive step reduction.

**Mechanism:** CFG introduces additional curvature into sampling trajectories (see Section 8). This curvature requires smaller integration steps for accurate tracking; reducing steps can cause integration errors.

**Mitigation:**
- Use higher-order solvers (e.g., DPM-Solver [^lu2022dpmsolver])
- Guidance distillation (Section 9)
- Dynamic guidance that reduces $\gamma$ at later steps

### 7.7 Computational Cost

**Phenomenon:** CFG doubles inference cost.

**Mechanism:** Each sampling step requires two forward passes (conditional and unconditional).

**Impact:** For large models (billions of parameters) and real-time applications, this overhead may be prohibitive.

**Mitigation:**
- Guidance distillation
- Batched inference (see Section 12)
- Early stopping of unconditional computation

---

## 8. Interaction with Rectified Flows

Rectified flows [^liu2023rectified][^liu2023instaflow] represent a complementary technique that aims to linearize the transport map, enabling efficient few-step sampling. Understanding the interaction between CFG and rectified flows is essential for deploying both techniques together.

### 8.1 Background: Rectified Flows

**Definition 8.1 (Rectified Flow).** *A rectified flow is obtained by iteratively "straightening" the learned velocity field such that trajectories approximate straight-line paths from prior to data.*

The rectification procedure trains a new model to match the marginal distributions of a pre-trained model while minimizing trajectory curvature:

$$\mathcal{L}_{\text{rect}} = \mathbb{E}_{t, x_0, x_1}\left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

where $(x_0, x_1)$ are coupled samples from the prior and data distributions.

![Rectified Flow](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/visualizations/rectified_flow_trajectory_curvature.gif?raw=true)
*Figure 12: Trajectory curvature under rectified flow. The straightening procedure reduces trajectory curvature, enabling accurate integration with fewer discretization steps.*

### 8.2 The Fundamental Tension

**Observation 8.1.** *Rectification and CFG have opposing effects on trajectory geometry:*

- **Rectification** minimizes trajectory curvature, enabling straight-line paths
- **CFG** introduces additional curvature through the guidance term

To see this, note that the CFG velocity field is:

$$v_{\text{cfg}}(x_t, t) = v_{\text{uncond}}(x_t, t) + \gamma (v_{\text{cond}}(x_t, t) - v_{\text{uncond}}(x_t, t))$$

Even if both $v_{\text{cond}}$ and $v_{\text{uncond}}$ are rectified (straight), their weighted combination is generally *not* straight unless the conditional and unconditional flows are parallel—which is typically not the case.

### 8.3 Quantifying the Curvature

The curvature of a trajectory $x(t)$ can be measured by:

$$\kappa(t) = \frac{\| \ddot{x}(t) \|}{\| \dot{x}(t) \|^2}$$

For CFG trajectories:
$$\dot{x} = v_{\text{cfg}} = v_u + \gamma(v_c - v_u)$$

The acceleration includes terms from both the time evolution of the individual velocity fields and their interaction:

$$\ddot{x} = \frac{\partial v_{\text{cfg}}}{\partial t} + (v_{\text{cfg}} \cdot \nabla) v_{\text{cfg}}$$

High guidance scales amplify the $(v_c - v_u)$ term, which generally has non-zero curvature, increasing overall trajectory curvature.

### 8.4 Empirical Evidence

#### Non-Overlapping Classes

![Rectified CFG Trajectory Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/non_overlapping/visualizations/rectified_cfg_trajectory.gif?raw=true)
*Figure 13: Trajectory evolution under rectified CFG for non-overlapping class distributions. Despite rectification of the base models, guidance introduces non-trivial curvature, particularly in the latter stages of sampling where trajectories diverge toward distinct class modes.*

![Rectified CFG Probability Path Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/non_overlapping/visualizations/rectified_cfg_probability_path_rect.gif?raw=true)
*Figure 14: Probability density evolution under rectified CFG (non-overlapping classes). The transport of probability mass exhibits greater complexity than unguided rectified flow.*

#### Overlapping Classes

![Rectified CFG Trajectory Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/overlapping/visualizations/rectified_cfg_trajectory.gif?raw=true)
*Figure 15: Trajectory evolution under rectified CFG for overlapping class distributions. The interaction between guidance and class overlap produces particularly complex trajectory dynamics in regions of distributional ambiguity.*

![Rectified CFG Probability Path Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/overlapping/visualizations/rectified_cfg_probability_path_rect.gif?raw=true)
*Figure 16: Probability density evolution under rectified CFG (overlapping classes). Note the intricate splitting of probability mass as trajectories navigate between overlapping modes.*

### 8.5 Implications for Few-Step Sampling

The curvature introduced by CFG has direct implications for step efficiency:

**Proposition 8.1.** *Let $\epsilon$ denote the integration error per step for a first-order solver. For rectified flow without CFG, the error scales as $\epsilon \propto h^2 \kappa$, where $h$ is the step size and $\kappa$ is the trajectory curvature. With CFG, the effective curvature increases to $\kappa_{\text{cfg}} > \kappa$, requiring proportionally smaller steps for equivalent accuracy.*

**Practical implication:** Models optimized for 1-4 step generation (e.g., InstaFlow [^liu2023instaflow], SDXL Turbo [^sauer2023adversarial]) may require modified guidance strategies:

1. **Reduced guidance scales**: Use $\gamma \in [1, 3]$ instead of $\gamma \in [5, 15]$
2. **Guidance distillation**: Train the model to directly produce CFG-equivalent outputs
3. **Adaptive guidance**: Reduce $\gamma$ as the step count decreases

---

## 9. Guidance Distillation

Given the computational overhead and geometric complications of CFG, a natural question is whether the guidance effect can be *distilled* into a single model.

### 9.1 Motivation

Guidance distillation addresses several limitations of standard CFG:

1. **Computational cost**: Eliminates the need for two forward passes
2. **Curvature**: The distilled model learns the guided velocity field directly, potentially with lower curvature
3. **Step efficiency**: Enables combination with rectified flows and few-step sampling

### 9.2 Methodology

The distillation procedure, introduced by Meng et al. [^meng2023distillation], trains a student model $v_\phi$ to match the CFG-modified teacher outputs:

$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{t, x_t, c}\left[ \| v_\phi(x_t, t, c) - v_{\text{cfg}}(x_t, t, c; \gamma) \|^2 \right]$$

where:
$$v_{\text{cfg}}(x_t, t, c; \gamma) = v_\theta(x_t, t, \varnothing) + \gamma (v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing))$$

The student model $v_\phi$ takes only the conditional input $c$ and produces outputs equivalent to CFG with scale $\gamma$.

**Remark 9.1.** *Distillation can be performed for a fixed guidance scale $\gamma$, or the scale can be provided as an additional input to enable variable guidance at inference time.*

### 9.3 Variants

**Progressive distillation** [^salimans2022progressive]: Combines guidance distillation with step distillation, iteratively halving the number of sampling steps while maintaining quality.

**Consistency distillation** [^song2023consistency]: Distills the entire sampling trajectory into a single-step model while incorporating CFG.

**Adversarial distillation** [^sauer2023adversarial]: Uses adversarial training in addition to distillation losses, enabling high-quality single-step generation with implicit guidance.

### 9.4 Empirical Results

#### Non-Overlapping Classes

![Distillation Guidance Trajectory Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/non_overlapping/distilled/visualizations/trajectory_guidance_scales.gif?raw=true)
*Figure 17: Trajectories from a guidance-distilled model (non-overlapping classes). The distilled model produces CFG-equivalent outputs with a single forward pass. Note the smooth, curved trajectories that directly encode the guidance effect.*

![Distillation Guidance Probability Path Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/non_overlapping/distilled/visualizations/probability_path_guidance_scales.gif?raw=true)
*Figure 18: Probability path under guidance distillation (non-overlapping classes). The probability transport closely matches the CFG trajectories shown in earlier figures.*

#### Overlapping Classes

![Distillation Guidance Trajectory Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/overlapping/distilled/visualizations/trajectory_guidance_scales.gif?raw=true)
*Figure 19: Trajectories from a guidance-distilled model (overlapping classes). The distilled model handles distributional overlap similarly to explicit CFG.*

![Distillation Guidance Probability Path Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/overlapping/distilled/visualizations/probability_path_guidance_scales.gif?raw=true)
*Figure 20: Probability path under guidance distillation (overlapping classes).*

### 9.5 Tradeoffs

| Aspect | Explicit CFG | Guidance Distillation |
|--------|--------------|----------------------|
| **Computational cost** | 2× forward passes | 1× forward pass |
| **Flexibility** | Variable $\gamma$ at inference | Fixed $\gamma$ (unless parameterized) |
| **Training cost** | Standard | Additional distillation phase |
| **Quality** | Reference | Slight degradation possible |
| **Compatibility** | All models | Requires distillation training |

---

## 10. Alternatives to CFG

While CFG has become the dominant guidance technique, several alternatives exist that may be preferable in specific contexts.

### 10.1 Classifier Guidance (Revisited)

Despite CFG's advantages, classifier guidance retains utility in certain scenarios:

- **Post-hoc control**: When the generative model is fixed and cannot be retrained with label dropout
- **Fine-grained attributes**: When guidance requires attribute-level control not captured in the training conditioning
- **Multimodal guidance**: Combining multiple pre-trained classifiers for different attributes

### 10.2 CLIP Guidance

CLIP guidance [^nichol2022glide] uses the CLIP model to provide text-based guidance:

$$\nabla_x \cos(E_{\text{image}}(x_t), E_{\text{text}}(\text{prompt}))$$

**Advantages:**
- Zero-shot text conditioning without text-image paired training
- Applicable to models trained without text conditioning

**Disadvantages:**
- Computationally expensive (backprop through CLIP)
- Can produce adversarial artifacts
- Semantic alignment may be imperfect

### 10.3 Self-Guidance

Self-guidance [^hong2023selfguidance] exploits internal representations of the diffusion model:

$$\nabla_x f(h_l(x_t))$$

where $h_l(x_t)$ is an intermediate activation and $f$ is an objective function (e.g., attention to a specific region).

**Advantages:**
- No external models required
- Enables spatial and structural control
- Zero additional training

**Disadvantages:**
- Limited to properties captured in internal representations
- Requires architectural understanding for effective use

### 10.4 Autoguidance

Autoguidance [^karras2024guiding] uses a smaller model as the "unconditional" baseline:

$$v_{\text{auto}} = v_{\text{large}}(x_t, t, c) + \gamma (v_{\text{large}}(x_t, t, c) - v_{\text{small}}(x_t, t, c))$$

**Intuition:** The smaller model captures less conditional information, serving as a natural baseline for the "unconditional" direction.

**Advantages:**
- More principled baseline than null conditioning
- Can improve quality beyond standard CFG

**Disadvantages:**
- Requires training multiple model sizes
- Still requires two forward passes

### 10.5 Negative Prompting

Negative prompting modifies the CFG formula to steer *away* from undesired attributes:

$$v_{\text{neg}} = v_{\text{uncond}} + \gamma_+ (v_{\text{pos}} - v_{\text{uncond}}) - \gamma_- (v_{\text{neg}} - v_{\text{uncond}})$$

where $v_{\text{pos}}$ and $v_{\text{neg}}$ are velocities conditioned on positive and negative prompts, respectively.

**Applications:**
- Removing artifacts ("blurry, low quality")
- Avoiding specific content ("no text, no watermark")
- Style steering ("not photorealistic")

**Limitations:**
- Three forward passes required
- Interaction between positive and negative guidance can be complex

### 10.6 Perp-Neg

Perp-Neg [^armandpour2023perpneg] orthogonalizes the negative prompt direction to avoid interference:

$$v_{\text{neg}}^{\perp} = v_{\text{neg}} - \frac{v_{\text{neg}} \cdot v_{\text{pos}}}{\|v_{\text{pos}}\|^2} v_{\text{pos}}$$

This ensures the negative guidance does not inadvertently suppress desired attributes.

### 10.7 Comparison Table

| Method | Forward Passes | External Models | Flexibility | Typical Use Case |
|--------|---------------|-----------------|-------------|------------------|
| **CFG** | 2 | No | High | General conditional generation |
| **Classifier** | 2+ | Classifier | Medium | Class-specific guidance |
| **CLIP** | 1+ | CLIP | High | Zero-shot text guidance |
| **Self-guidance** | 1 | No | Medium | Spatial/structural control |
| **Autoguidance** | 2 | Smaller model | High | Quality improvement |
| **Negative prompt** | 3 | No | High | Artifact removal |

---

## 11. Theoretical Limitations and Assumptions

A rigorous treatment of CFG must acknowledge the assumptions underlying its derivation and the conditions under which these assumptions may fail.

### 11.1 Score Approximation Quality

**Assumption:** The model $v_\theta$ accurately approximates the true conditional and unconditional score functions.

**Failure mode:** In regions of low data density, both scores may be poorly estimated. The CFG difference $v_{\text{cond}} - v_{\text{uncond}}$ may then be dominated by estimation noise rather than meaningful guidance signal.

**Consequence:** High guidance scales can amplify these estimation errors, producing artifacts in low-density regions.

### 11.2 Label Dropout Sufficiency

**Assumption:** Training with label dropout (e.g., $p_{\text{dropout}} = 0.1$) enables the model to learn both conditional and unconditional distributions adequately.

**Failure mode:** The unconditional distribution may be underrepresented in training, leading to poor unconditional velocity estimates.

**Consequence:** The "unconditional" velocity $v_{\text{uncond}}$ may not accurately represent $\nabla_x \log p(x)$, distorting the implicit classifier direction.

### 11.3 Geometric Assumptions

**Assumption:** The conditional and unconditional distributions share sufficient geometric structure for their score difference to be meaningful.

**Failure mode:** If the conditional distribution is concentrated on a low-dimensional manifold disjoint from the unconditional support, the guidance direction may be ill-defined.

### 11.4 Static Guidance Scale

**Assumption:** A single guidance scale $\gamma$ is appropriate across all timesteps and spatial locations.

**Failure mode:** Early in sampling (high noise), strong guidance may be beneficial for establishing global structure. Late in sampling (low noise), it may introduce artifacts by over-sharpening details.

**Mitigation:** Adaptive guidance schedules (Section 6.3).

### 11.5 Independence of Conditions

**Assumption:** For multi-condition prompts, the conditions can be combined linearly.

**Failure mode:** Compositional prompts with conflicting or dependent conditions may not admit a single guidance direction that satisfies all constraints.

---

## 12. Practical Considerations

### 12.1 Computational Cost

The primary computational cost of CFG is the requirement for two forward passes per sampling step:

$$\text{Cost}_{\text{CFG}} = 2 \times \text{Cost}_{\text{conditional}}$$

For large models (e.g., Stable Diffusion XL with ~3.5B parameters), this overhead is significant:

| Model | Parameters | Single Pass (A100) | CFG (2 passes) |
|-------|------------|-------------------|----------------|
| SD 1.5 | 860M | ~50ms | ~100ms |
| SD 2.1 | 865M | ~55ms | ~110ms |
| SDXL | 3.5B | ~150ms | ~300ms |
| SD 3 | 8B | ~300ms | ~600ms |

### 12.2 Memory Implications

CFG does not necessarily double memory requirements if implemented efficiently:

**Sequential computation:**
```python
v_uncond = model(x_t, t, null_cond)
v_cond = model(x_t, t, cond)
v_cfg = v_uncond + gamma * (v_cond - v_uncond)
```

This approach performs two sequential forward passes, requiring memory for only one activation set at a time.

**Batched computation:**
```python
x_batch = torch.cat([x_t, x_t], dim=0)
c_batch = torch.cat([null_cond, cond], dim=0)
v_batch = model(x_batch, t, c_batch)
v_uncond, v_cond = v_batch.chunk(2, dim=0)
v_cfg = v_uncond + gamma * (v_cond - v_uncond)
```

Batched computation can be more efficient on GPUs due to better parallelization, but requires 2× memory for activations.

### 12.3 Batching Strategies

For multi-sample generation, several batching strategies are possible:

**Strategy 1: Full batching**
- Batch all conditional and unconditional computations together
- Memory: $2 \times N \times \text{activation\_size}$
- Throughput: Optimal for GPU utilization

**Strategy 2: Split batching**
- Compute conditional batch, then unconditional batch
- Memory: $N \times \text{activation\_size}$
- Throughput: ~2× wall time

**Strategy 3: Interleaved**
- Alternate between samples
- Memory: Minimal
- Throughput: Poor GPU utilization

**Recommendation:** Use full batching when memory permits; split batching otherwise.

### 12.4 Gradient Checkpointing

For memory-constrained settings, gradient checkpointing can reduce activation memory at the cost of additional computation:

```python
from torch.utils.checkpoint import checkpoint

def cfg_step(model, x_t, t, cond, null_cond, gamma):
    v_uncond = checkpoint(model, x_t, t, null_cond)
    v_cond = checkpoint(model, x_t, t, cond)
    return v_uncond + gamma * (v_cond - v_uncond)
```

### 12.5 Half-Precision and Quantization

CFG is compatible with standard optimization techniques:

- **FP16/BF16**: Both forward passes can use mixed precision
- **Quantization**: 8-bit and 4-bit models work with CFG
- **Attention optimization**: Flash attention, xformers compatible

### 12.6 Caching Optimizations

For static conditioning (e.g., text embeddings), the conditioning computation can be cached:

```python
# Precompute text embeddings (once)
cond_embed = text_encoder(prompt)
null_embed = text_encoder("")

# Sampling loop (many steps)
for t in timesteps:
    # Embeddings are reused
    v_cfg = cfg_step(model, x_t, t, cond_embed, null_embed, gamma)
    x_t = euler_step(x_t, v_cfg, dt)
```

---

## 13. Conclusion

### 13.1 Summary of Key Insights

This work has presented a comprehensive treatment of classifier-free guidance, yielding several key insights:

1. **Theoretical foundation:** CFG is mathematically equivalent to classifier guidance with an implicit classifier derived from the generative model itself. The guided distribution is a geometric mixture of conditional and unconditional distributions.

2. **Practical utility:** CFG provides a simple, effective mechanism for improving conditional sample quality across diverse domains. Typical guidance scales range from 2-15 depending on the application.

3. **Failure modes:** CFG can produce oversaturation, mode collapse, and compositional failures at high guidance scales. It may also conflict with few-step sampling methods like rectified flows.

4. **Alternatives:** Classifier guidance, CLIP guidance, self-guidance, autoguidance, and negative prompting offer complementary capabilities for specific use cases.

5. **Computational considerations:** CFG doubles inference cost. Batching strategies, guidance distillation, and caching can mitigate this overhead.

### 13.2 Open Questions

Several fundamental questions remain open:

1. **Optimal guidance schedules:** Is there a principled way to derive time-dependent guidance scales from first principles?

2. **Compositional guidance:** How can CFG be extended to handle complex, multi-attribute conditions more effectively?

3. **Theoretical guarantees:** Under what conditions does CFG provably improve sample quality? Can the quality-diversity tradeoff be characterized formally?

4. **Implicit classifier properties:** What properties does the implicit classifier $\nabla_x \log p(c|x_t)$ inherit from the generative model? How does it compare to explicitly trained classifiers?

5. **Interaction effects:** How do CFG, rectification, and distillation interact? Can these techniques be unified in a principled framework?

### 13.3 Future Directions

Several promising directions emerge from this analysis:

1. **Learned guidance:** Training models to predict optimal guidance scales or directions, rather than using fixed formulas.

2. **Hierarchical guidance:** Applying different guidance strategies at different scales or layers of the model.

3. **Uncertainty-aware guidance:** Modulating guidance strength based on model uncertainty in different regions of the sample space.

4. **Efficient alternatives:** Developing guidance methods that achieve CFG-like benefits without doubling computational cost.

5. **Formal analysis:** Establishing theoretical frameworks that explain when and why CFG improves sample quality.

### 13.4 Recommendations for Practitioners

Based on this analysis, the following recommendations are offered:

1. **Start with domain defaults** (Table 1) and tune guidance scale based on quality-diversity tradeoff.

2. **Monitor for oversaturation** at high guidance scales; reduce $\gamma$ or use CFG rescaling if artifacts appear.

3. **Consider guidance distillation** for deployment scenarios where computational cost is critical.

4. **Use adaptive schedules** when combining CFG with few-step sampling.

5. **Employ negative prompting** judiciously for artifact removal, but be aware of increased computational cost.

The visualizations and analyses presented herein suggest that the interaction between CFG, distributional structure, and trajectory geometry is richer than current theory captures. A deeper mathematical understanding of these phenomena remains an important open problem with significant practical implications.

---

## References

[^ho2022classifier]: Ho, J. and Salimans, T. "Classifier-Free Diffusion Guidance." *arXiv preprint arXiv:2207.12598*, 2022.

[^dhariwal2021diffusion]: Dhariwal, P. and Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." In *Advances in Neural Information Processing Systems* 34, pp. 8780-8794, 2021.

[^ho2020denoising]: Ho, J., Jain, A., and Abbeel, P. "Denoising Diffusion Probabilistic Models." In *Advances in Neural Information Processing Systems* 33, pp. 6840-6851, 2020.

[^song2019generative]: Song, Y. and Ermon, S. "Generative Modeling by Estimating Gradients of the Data Distribution." In *Advances in Neural Information Processing Systems* 32, 2019.

[^song2021scorebased]: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., and Poole, B. "Score-Based Generative Modeling through Stochastic Differential Equations." In *International Conference on Learning Representations*, 2021.

[^sohl2015deep]: Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." In *International Conference on Machine Learning*, pp. 2256-2265, 2015.

[^lipman2023flow]: Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., and Le, M. "Flow Matching for Generative Modeling." In *International Conference on Learning Representations*, 2023.

[^liu2023flow]: Liu, X., Gong, C., and Liu, Q. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." In *International Conference on Learning Representations*, 2023.

[^liu2023rectified]: Liu, X., Gong, C., and Liu, Q. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." In *International Conference on Learning Representations*, 2023.

[^liu2023instaflow]: Liu, X., Zhang, X., Ma, J., Peng, J., and Liu, Q. "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation." In *International Conference on Learning Representations*, 2024.

[^ramesh2022hierarchical]: Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. "Hierarchical Text-Conditional Image Generation with CLIP Latents." *arXiv preprint arXiv:2204.06125*, 2022.

[^rombach2022highresolution]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models." In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 10684-10695, 2022.

[^saharia2022photorealistic]: Saharia, C., Chan, W., Saxena, S., et al. "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." In *Advances in Neural Information Processing Systems* 35, 2022.

[^nichol2022glide]: Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., and Chen, M. "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models." In *International Conference on Machine Learning*, pp. 16784-16804, 2022.

[^ho2022video]: Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D.J. "Video Diffusion Models." In *Advances in Neural Information Processing Systems* 35, 2022.

[^poole2023dreamfusion]: Poole, B., Jain, A., Barron, J.T., and Mildenhall, B. "DreamFusion: Text-to-3D using 2D Diffusion." In *International Conference on Learning Representations*, 2023.

[^kong2021diffwave]: Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. "DiffWave: A Versatile Diffusion Model for Audio Synthesis." In *International Conference on Learning Representations*, 2021.

[^liu2023audioldm]: Liu, H., Chen, Z., Yuan, Y., Mei, X., Liu, X., Mandic, D., Wang, W., and Plumbley, M.D. "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models." In *International Conference on Machine Learning*, 2023.

[^hong2023selfguidance]: Hong, S., Lee, G., Cho, W., and Kim, S. "Improving Sample Quality of Diffusion Models Using Self-Attention Guidance." In *International Conference on Computer Vision*, 2023.

[^karras2024analyzing]: Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., and Laine, S. "Analyzing and Improving the Training Dynamics of Diffusion Models." In *Computer Vision and Pattern Recognition*, 2024.

[^karras2024guiding]: Karras, T., Aittala, M., Kynkäänniemi, T., et al. "Guiding a Diffusion Model with a Bad Version of Itself." *arXiv preprint arXiv:2406.02507*, 2024.

[^kynkaanniemi2024applying]: Kynkäänniemi, T., Karras, T., Aittala, M., Aila, T., and Lehtinen, J. "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models." *arXiv preprint arXiv:2404.07724*, 2024.

[^meng2023distillation]: Meng, C., Rombach, R., Gao, R., Kingma, D.P., Ermon, S., Ho, J., and Salimans, T. "On Distillation of Guided Diffusion Models." In *Computer Vision and Pattern Recognition*, pp. 14297-14306, 2023.

[^salimans2022progressive]: Salimans, T. and Ho, J. "Progressive Distillation for Fast Sampling of Diffusion Models." In *International Conference on Learning Representations*, 2022.

[^song2023consistency]: Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. "Consistency Models." In *International Conference on Machine Learning*, 2023.

[^sauer2023adversarial]: Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. "Adversarial Diffusion Distillation." *arXiv preprint arXiv:2311.17042*, 2023.

[^liu2023compositional]: Liu, N., Li, S., Du, Y., Torralba, A., and Tenenbaum, J.B. "Compositional Visual Generation with Composable Diffusion Models." In *European Conference on Computer Vision*, pp. 423-439, 2022.

[^hertz2023prompttoprompt]: Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., and Cohen-Or, D. "Prompt-to-Prompt Image Editing with Cross-Attention Control." In *International Conference on Learning Representations*, 2023.

[^lin2024common]: Lin, S., Liu, B., Li, J., and Yang, X. "Common Diffusion Noise Schedules and Sample Steps are Flawed." In *IEEE/CVF Winter Conference on Applications of Computer Vision*, 2024.

[^lu2022dpmsolver]: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." In *Advances in Neural Information Processing Systems* 35, 2022.

[^goodfellow2015explaining]: Goodfellow, I.J., Shlens, J., and Szegedy, C. "Explaining and Harnessing Adversarial Examples." In *International Conference on Learning Representations*, 2015.

[^graikos2023diffusion]: Graikos, A., Malkin, N., Jojic, N., and Samaras, D. "Diffusion Models as Plug-and-Play Priors." In *Advances in Neural Information Processing Systems* 35, 2022.

[^mirza2014conditional]: Mirza, M. and Osindero, S. "Conditional Generative Adversarial Nets." *arXiv preprint arXiv:1411.1784*, 2014.

[^brock2019large]: Brock, A., Donahue, J., and Simonyan, K. "Large Scale GAN Training for High Fidelity Natural Image Synthesis." In *International Conference on Learning Representations*, 2019.

[^du2019implicit]: Du, Y. and Mordatch, I. "Implicit Generation and Generalization in Energy-Based Models." In *Advances in Neural Information Processing Systems* 32, 2019.

[^armandpour2023perpneg]: Armandpour, M., Zheng, H., Sadeghian, A., Sber, A., and Zhou, M. "Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, Alleviate Janus Problem and Beyond." *arXiv preprint arXiv:2304.04968*, 2023.
