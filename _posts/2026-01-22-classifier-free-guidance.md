---
layout: post
title: "Classifier-Free Guidance [WIP]"
date: 2026-01-22
#categories: [machine-learning, generative-models, diffusion-models, flow-matching]
tags: [classifier-free-guidance, sampling, generative-models, diffusion-models, flow-matching]
math: true
excerpt: "An overview of classifier-free guidance for generative models."
---

# Introduction

Classifier-free guidance is a technique for improving the quality of generated samples from generative models. It has become a staple in diffusion models and flow matching for conditional generation that it is almost always used. This post takes a deep dive into the topic giving a visual explainer and at the same time trying to answer questions like: why do we run the model twice? is it just temperature tuning? and when should we not use it?

![CFG Trajectory Curvature](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/cfg/visualizations/cfg_vector_field.gif?raw=true)

# Background

## Diffusion Models / Flow Matching

Both diffusion models and flow matching can be interpreted as utilizing the Langevin dynamics, specifically the Metropolis-adjusted Langevin algorithm (MALA), to sample from the target distribution. MALA is given by the following equation:

$$x_{t+1} = x_t + \alpha_t \nabla_x \log p(x_t) + \mathcal{N}(0, \epsilon_t^2)$$

where $x_t$ is the sample at time $t$, $\nabla_x \log p(x_t)$ is the gradient of the log-likelihood of the target distribution, $\alpha_t$ is the step size, and $\mathcal{N}(0, \epsilon_t^2)$ is the noise added to the sample.

While diffusion models learn to predict the score function, i.e. one component of $\nabla_x \log p(x_t)$, flow matching learns to predict the velocity field itself, i.e. $v_t = x_{t+1} - x_{t}$. The diffusion models are learning to denoise the noisy sample, while flow matching learns the linear interpolation between the current sample and the next one. 

For simplicity, we will assume a flow matching model for the rest of the post.

![Diffusion/Flow Model Visualization](/images/diffusion_flow_visual.svg)

*Figure 1: (a) A single forward pass through the model takes noisy input x_t and timestep t, producing a velocity/score prediction. (b) Iterative sampling follows these velocity predictions step-by-step from noise distribution to data distribution. (c) The model learns a vector field that transports samples from a Gaussian distribution $\mathcal{N}(0, 1)$ to the target distribution $p(\mathbf{x})$*

---
    WIP 
---

## Conditional vs Unconditional Generation
- What conditioning means (class label, text prompt, etc.)
- The quality-diversity tradeoff: unconditional = diverse but unfocused; conditional = focused but can mode-collapse

## Classifier Guidance
- The original approach: train external classifier on noisy data
- Use classifier gradients to steer generation
- The bottleneck: requires separate classifier, noisy-data training, limited flexibility

# Classifier-Free Guidance

The key insight is to use a classifier to guide the sampling process. The classifier is trained to predict the class of the generated sample, and the guidance is used to improve the sample quality. The guidance is given by the following formula:

$$x = x_{uncond} + \text{cfg_scale} \cdot (x_{cond} - x_{uncond})$$

where $x_{uncond}$ is the generated sample without the class label, and $x_{cond}$ is the generated sample with the class label.
The guidance scale $\text{cfg_scale}$ is a hyperparameter that controls the strength of the guidance. A higher guidance scale will produce more class-specific samples.

## The Algorithm

The algorithm is as follows:

1. Train a classifier to predict the class of the generated sample.
2. Use the classifier to guide the sampling process.
3. Improve the sample quality.

## The Implementation

The code for the classifier-free guidance is as follows:

```python
def classifier_free_guidance(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, cfg_scale: float = 2.0) -> torch.Tensor:
    """
    Args:
        model: The model to use for generation.
        x: The input tensor.
        y: The class tensor.
        cfg_scale: The guidance scale.

    Returns:
        The guided sample.
    """
    x_uncond = model(x, torch.tensor(-1.0))
    x_cond = model(x, y)
    x = x_uncond + cfg_scale * (x_cond - x_uncond)
    return x
```

## Temperature Tuning

Is classifier-free guidance a form of temperature tuning? The answer is yes and no.

Yes, because the guidance scale $\text{cfg_scale}$ > $1.0$ can be interpreted as a way to reduce the temperature of the sampling process.
No, because the outcome is a condensation of the distribution towards the class label. It is not a simple temperature reduction. Decreasing the $\text{cfg_scale}$ will not always result in a sharper distribution. In fact, it will result in a more blurred distribution.