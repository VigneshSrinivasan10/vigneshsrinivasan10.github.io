---
layout: post
title: "Classifier-Free Guidance: A Rigorous Analysis of Conditional Steering in Generative Models"
date: 2026-01-22
tags: [classifier-free-guidance, sampling, generative-models, diffusion-models, flow-matching]
math: true
excerpt: "A formal treatment of classifier-free guidance for conditional generation in diffusion models and flow matching, with theoretical foundations, empirical visualizations, and analysis of limitations."
---

## Abstract

Classifier-free guidance (CFG) has emerged as a dominant technique for improving sample quality in conditional generative models, particularly diffusion models and flow matching frameworks. This work provides a rigorous mathematical treatment of CFG, situating it within the broader context of score-based generative modeling. The theoretical foundations are derived from Bayesian principles, and the relationships between CFG, classifier guidance, and temperature scaling are formally analyzed. Through systematic visualizations, the behavior of CFG under varying distributional assumptions is examined. Limitations and failure modes are discussed, and connections to recent advances in rectified flows and guidance distillation are established.

---

## 1. Introduction

Classifier-free guidance, introduced by Ho and Salimans [^ho2022classifier], has become a ubiquitous technique in conditional generative modeling. The method appears to substantially improve the perceptual quality and conditional fidelity of generated samples across diverse domains, including text-to-image synthesis [^ramesh2022hierarchical], audio generation, and video synthesis. Despite its widespread adoption, the theoretical underpinnings and failure modes of CFG merit careful examination.

This work presents a formal analysis of classifier-free guidance, addressing several questions that arise in practice: (1) Why does CFG require two forward passes through the model? (2) Under what conditions does CFG improve sample quality? (3) What is the relationship between CFG and temperature-based sampling? (4) When might CFG fail or produce undesirable artifacts?

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

Prior to classifier-free guidance, Dhariwal and Nichol [^dhariwal2021diffusion] proposed using an external classifier to guide the sampling process. Given a pre-trained classifier $p_\phi(c | x)$, the guided score is computed as:

$$\nabla_x \log p(x_t | c) = \nabla_x \log p(x_t) + \gamma \nabla_x \log p_\phi(c | x_t) \tag{2}$$

where $\gamma \geq 0$ is a guidance scale hyperparameter.

This approach, while effective, presents several limitations:

1. **Representational mismatch:** The classifier must be trained on noisy samples $x_t$ at all noise levels, which may require a non-trivial modification of standard classifier training protocols.

2. **Mode coverage:** An unconditionally trained generative model may exhibit poor coverage of low-density regions of the data manifold, potentially leading to mode collapse under guidance [^dhariwal2021diffusion].

3. **Gradient incompatibility:** The gradients $\nabla_x \log p(x_t)$ and $\nabla_x \log p_\phi(c | x_t)$ may operate at different scales or exhibit misaligned geometry, complicating the selection of an appropriate $\gamma$.

### 2.4 Conditional Generative Models

An alternative to post-hoc guidance is to train the generative model conditionally from the outset. Let $c$ denote the conditioning information (e.g., class label, text embedding). The conditional model learns to approximate:

$$v_\theta(x_t, t, c) \approx \mathbb{E}[v_t | x_t, c]$$

where the expectation is taken over the conditional flow.

![Conditional Generation Visualization](/images/classifier_free_guidance_visual.svg)

*Figure 4: Conditional generation with explicit conditioning. (a) The forward pass incorporates conditioning signal $c$ as an additional input. (b) Conditional sampling: with $c = \text{left eye}$, trajectories are steered toward the corresponding region of the data manifold.*

Dhariwal and Nichol [^dhariwal2021diffusion] observed that classifier guidance can augment even conditionally trained models:

$$\nabla_x \log p(x_t | c) + \gamma \nabla_x \log p_\phi(c | x_t)$$

The theoretical justification for this combination remains an open question in the literature. Nevertheless, empirical results suggest improved sample quality under certain conditions.

---

## 3. Classifier-Free Guidance

The central contribution of Ho and Salimans [^ho2022classifier] is the elimination of the external classifier through a Bayesian reformulation that enables implicit guidance.

### 3.1 Theoretical Derivation

The derivation proceeds from Bayes' theorem. Consider the joint distribution factorization:

$$p(x_t, c) = p(x_t | c) p(c) = p(c | x_t) p(x_t)$$

Taking logarithms:

$$\log p(x_t | c) + \log p(c) = \log p(c | x_t) + \log p(x_t)$$

Rearranging:

$$\log p(x_t | c) = \log p(c | x_t) + \log p(x_t) - \log p(c) \tag{3}$$

**Assumption 3.1 (Conditional Independence).** *It is assumed that $p(c)$ does not depend on $x_t$, which holds when $c$ is exogenous conditioning information.*

Under this assumption, taking the gradient with respect to $x_t$:

$$\nabla_{x_t} \log p(x_t | c) = \nabla_{x_t} \log p(c | x_t) + \nabla_{x_t} \log p(x_t) \tag{4}$$

Rearranging to isolate the classifier gradient:

$$\nabla_{x_t} \log p(c | x_t) = \nabla_{x_t} \log p(x_t | c) - \nabla_{x_t} \log p(x_t) \tag{5}$$

![CFG Vectors](/images/cfg_vectors.svg)

*Figure 5: Geometric interpretation of classifier-free guidance. The CFG velocity vector is computed as a weighted combination of conditional and unconditional predictions. The guidance scale $\gamma$ modulates the magnitude of the conditional correction term.*

Substituting Equation (5) into the classifier guidance formulation (Equation 2):

$$\nabla_x \log \tilde{p}(x_t | c) = \nabla_x \log p(x_t) + \gamma \left( \nabla_{x_t} \log p(x_t | c) - \nabla_{x_t} \log p(x_t) \right) \tag{6}$$

where $\tilde{p}(x_t | c)$ denotes the guided distribution. Simplifying:

$$\nabla_x \log \tilde{p}(x_t | c) = (1 - \gamma) \nabla_x \log p(x_t) + \gamma \nabla_{x_t} \log p(x_t | c) \tag{7}$$

In the flow matching parameterization, where $v_\theta \propto \nabla_x \log p$, this yields:

$$v_{\text{cfg}} = (1 - \gamma) v_{\text{uncond}} + \gamma \, v_{\text{cond}} = v_{\text{uncond}} + \gamma (v_{\text{cond}} - v_{\text{uncond}}) \tag{8}$$

where $v_{\text{uncond}} = v_\theta(x_t, t, \varnothing)$ is the unconditional velocity and $v_{\text{cond}} = v_\theta(x_t, t, c)$ is the conditional velocity.

**Remark 3.1.** *The guidance scale $\gamma$ interpolates between unconditional ($\gamma = 0$) and conditional ($\gamma = 1$) generation. Values $\gamma > 1$ extrapolate beyond the conditional distribution, which may improve perceptual quality at the cost of sample diversity.*

### 3.2 Joint Training Protocol

A key implementation detail is that a single model learns both conditional and unconditional distributions through stochastic label dropout during training.

**Definition 3.1 (Null Conditioning).** *Let $\varnothing$ denote a null conditioning token. During training, the conditioning label $c$ is replaced with $\varnothing$ with probability $p_{\text{dropout}}$.*

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

**Remark 3.2.** *The computational cost of CFG is approximately twice that of standard conditional sampling, as two forward passes are required per integration step. This may be prohibitive for large-scale models or real-time applications.*

---

## 4. Related Work

### 4.1 Alternative Guidance Mechanisms

Several alternatives to CFG have been proposed in the literature:

- **CLIP guidance** [^nichol2022glide]: Uses CLIP embeddings to guide generation toward text descriptions without explicit classifier training.

- **Perceptual guidance** [^ho2022video]: Employs perceptual loss functions to steer generation toward desired image properties.

- **Self-guidance** [^hong2023selfguidance]: Exploits intermediate representations within the diffusion model itself to provide guidance signals.

### 4.2 Guidance Strength Adaptation

The fixed guidance scale $\gamma$ may be suboptimal across timesteps or samples:

- **Dynamic guidance** [^karras2024analyzing]: Proposes timestep-dependent guidance schedules.

- **Adaptive guidance** [^castillo2023cfg++]: Adjusts $\gamma$ based on sample-specific characteristics.

### 4.3 Distillation Approaches

Recent work has sought to distill the two-pass CFG procedure into a single forward pass:

- **Guidance distillation** [^meng2023distillation]: Trains a student model to directly predict $v_{\text{cfg}}$.

- **Progressive distillation** [^salimans2022progressive]: Iteratively reduces the number of sampling steps while preserving CFG behavior.

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

**Proposition 5.1 (Informal).** *CFG with $\gamma > 1$ may be interpreted as anisotropic temperature reduction, where the "cooling" is concentrated along the direction $v_{\text{cond}} - v_{\text{uncond}}$ rather than uniformly across the sample space.*

![Temperature Scaling Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/temperature-scaling/outputs/flow_gaussians/non_overlapping/temperature_scaling/temperature_comparison.png?raw=true)
*Figure 10: Comparison of temperature scaling effects for non-overlapping classes. Unlike CFG, uniform temperature reduction does not preferentially concentrate samples around conditional modes.*

![Temperature Scaling Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/temperature-scaling/outputs/flow_gaussians/overlapping/temperature_scaling/temperature_comparison.png?raw=true)
*Figure 11: Temperature scaling comparison for overlapping classes. The distinction between CFG and temperature scaling is more pronounced when class distributions overlap significantly.*

**Key Observation:** While both CFG and temperature scaling reduce sample diversity, they operate through fundamentally different mechanisms. Temperature scaling uniformly contracts the distribution, whereas CFG induces a directed shift toward the conditional mode. Consequently, reducing $\gamma$ below 1 does not produce sharper distributions—rather, it may result in samples that interpolate between the conditional and unconditional distributions.

---

## 6. Interaction with Rectified Flows

Rectified flows [^liu2023rectified][^liu2023instaflow] represent a complementary technique that aims to linearize the transport map, thereby enabling efficient few-step sampling.

**Definition 6.1 (Rectified Flow).** *A rectified flow is obtained by iteratively "straightening" the learned velocity field such that trajectories approximate straight-line paths from prior to data.*

![Rectified Flow](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/visualizations/rectified_flow_trajectory_curvature.gif?raw=true)
*Figure 12: Trajectory curvature under rectified flow. The straightening procedure reduces trajectory curvature, enabling accurate integration with fewer discretization steps.*

### 6.1 Tension Between Rectification and Guidance

An important observation is that CFG and rectification may exhibit conflicting objectives:

- **Rectification** seeks to minimize trajectory curvature.
- **CFG** introduces additional curvature through the guidance term.

This tension suggests that naive application of CFG to rectified flows may partially undo the benefits of rectification.

### 6.2 Rectified CFG Trajectories

#### Non-Overlapping Classes

![Rectified CFG Trajectory Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/non_overlapping/visualizations/rectified_cfg_trajectory.gif?raw=true)
*Figure 13: Trajectory evolution under rectified CFG for non-overlapping class distributions. Despite rectification, guidance introduces non-trivial curvature in the latter stages of sampling.*

![Rectified CFG Probability Path Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/non_overlapping/visualizations/rectified_cfg_probability_path_rect.gif?raw=true)
*Figure 14: Probability density evolution under rectified CFG (non-overlapping classes).*

#### Overlapping Classes

![Rectified CFG Trajectory Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/overlapping/visualizations/rectified_cfg_trajectory.gif?raw=true)
*Figure 15: Trajectory evolution under rectified CFG for overlapping class distributions. The interaction between guidance and class overlap produces more complex trajectory dynamics.*

![Rectified CFG Probability Path Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/overlapping/visualizations/rectified_cfg_probability_path_rect.gif?raw=true)
*Figure 16: Probability density evolution under rectified CFG (overlapping classes).*

### 6.3 Guidance Distillation

One approach to reconciling CFG with few-step sampling is guidance distillation, wherein a student model is trained to directly predict the guided velocity field.

#### Non-Overlapping Classes

![Distillation Guidance Trajectory Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/non_overlapping/distilled/visualizations/trajectory_guidance_scales.gif?raw=true)
*Figure 17: Trajectories from a guidance-distilled model (non-overlapping classes). The distilled model produces CFG-equivalent outputs with a single forward pass.*

![Distillation Guidance Probability Path Non-Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/non_overlapping/distilled/visualizations/probability_path_guidance_scales.gif?raw=true)
*Figure 18: Probability path under guidance distillation (non-overlapping classes).*

#### Overlapping Classes

![Distillation Guidance Trajectory Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/overlapping/distilled/visualizations/trajectory_guidance_scales.gif?raw=true)
*Figure 19: Trajectories from a guidance-distilled model (overlapping classes).*

![Distillation Guidance Probability Path Overlapping](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/main/outputs/flow_gaussians/overlapping/distilled/visualizations/probability_path_guidance_scales.gif?raw=true)
*Figure 20: Probability path under guidance distillation (overlapping classes).*

---

## 7. Limitations and Failure Modes

Several limitations of CFG should be acknowledged:

### 7.1 Theoretical Assumptions

1. **Score approximation quality:** The derivation assumes that $v_\theta$ accurately approximates the true score function. In practice, approximation errors may compound under guidance, particularly for large $\gamma$.

2. **Label dropout sufficiency:** The joint training procedure assumes that unconditional and conditional distributions can be adequately learned through label dropout alone. This may not hold for complex, multi-modal conditioning.

3. **Continuous guidance:** The formulation assumes $\gamma$ is fixed throughout sampling. Optimal guidance may vary with timestep or sample region.

### 7.2 Practical Failure Modes

1. **Oversaturation:** Large guidance scales ($\gamma \gg 1$) may produce oversaturated or unnatural samples, particularly in image generation [^karras2024analyzing].

2. **Mode collapse:** Aggressive guidance can reduce diversity and collapse to a small number of prototypical samples.

3. **Compositional failures:** CFG may struggle with compositional prompts (e.g., "a red cube and a blue sphere") where multiple conditions must be simultaneously satisfied [^liu2023compositional].

4. **Out-of-distribution conditions:** When the conditioning signal is far from the training distribution, the guidance direction $v_{\text{cond}} - v_{\text{uncond}}$ may be unreliable.

### 7.3 Computational Considerations

The requirement for two forward passes per sampling step doubles the computational cost relative to unguided generation. For large-scale models (e.g., multi-billion parameter networks), this overhead is substantial.

---

## 8. Conclusion

This work has presented a rigorous treatment of classifier-free guidance, deriving its mathematical foundations from Bayesian principles and examining its empirical behavior across distributional regimes. While CFG has demonstrated substantial utility in conditional generative modeling, the analysis reveals important limitations and failure modes that merit consideration in practical applications.

Several directions for future work emerge from this analysis:

1. **Adaptive guidance schedules:** Developing principled methods for varying $\gamma$ across timesteps or samples.

2. **Efficient CFG approximations:** Reducing the computational overhead through distillation or amortization.

3. **Theoretical guarantees:** Establishing formal conditions under which CFG provably improves sample quality.

4. **Compositional guidance:** Extending CFG to handle complex, multi-attribute conditioning.

The visualizations presented herein suggest that the interaction between CFG, distributional overlap, and trajectory rectification is richer than current theory captures. A deeper mathematical understanding of these phenomena remains an important open problem.

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

[^liu2023rectified]: Liu, X., Zhang, X., Ma, J., Peng, J., and Liu, Q. "Instaflow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation." In *International Conference on Learning Representations*, 2024.

[^liu2023instaflow]: Liu, X., Zhang, X., Ma, J., Peng, J., and Liu, Q. "Instaflow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation." *arXiv preprint arXiv:2309.06380*, 2023.

[^ramesh2022hierarchical]: Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. "Hierarchical Text-Conditional Image Generation with CLIP Latents." *arXiv preprint arXiv:2204.06125*, 2022.

[^nichol2022glide]: Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., and Chen, M. "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models." In *International Conference on Machine Learning*, pp. 16784-16804, 2022.

[^ho2022video]: Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D.J. "Video Diffusion Models." In *Advances in Neural Information Processing Systems* 35, 2022.

[^hong2023selfguidance]: Hong, S., Lee, G., Cho, W., and Kim, S. "Improving Sample Quality of Diffusion Models Using Self-Attention Guidance." In *International Conference on Computer Vision*, 2023.

[^karras2024analyzing]: Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., and Laine, S. "Analyzing and Improving the Training Dynamics of Diffusion Models." In *Computer Vision and Pattern Recognition*, 2024.

[^castillo2023cfg++]: Chung, H., Kim, J., Kim, S., and Ye, J.C. "CFG++: Manifold-Constrained Classifier Free Guidance for Diffusion Models." *arXiv preprint arXiv:2406.08070*, 2024.

[^meng2023distillation]: Meng, C., Rombach, R., Gao, R., Kingma, D.P., Ermon, S., Ho, J., and Salimans, T. "On Distillation of Guided Diffusion Models." In *Computer Vision and Pattern Recognition*, pp. 14297-14306, 2023.

[^salimans2022progressive]: Salimans, T. and Ho, J. "Progressive Distillation for Fast Sampling of Diffusion Models." In *International Conference on Learning Representations*, 2022.

[^liu2023compositional]: Liu, N., Li, S., Du, Y., Torralba, A., and Tenenbaum, J.B. "Compositional Visual Generation with Composable Diffusion Models." In *European Conference on Computer Vision*, pp. 423-439, 2022.
