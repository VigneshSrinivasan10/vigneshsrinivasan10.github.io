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

![CFG Trajectory Curvature](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/visualizations/cfg_vector_field.gif?raw=true)  

# Background

## Diffusion Models / Flow Matching

Both diffusion models and flow matching can be interpreted as utilizing the Langevin dynamics, specifically the Metropolis-adjusted Langevin algorithm (MALA), to sample from the target distribution. MALA is given by the following equation:

$$x_{t+1} = x_t + \alpha_t \nabla_x \log p(x_t) + \mathcal{N}(0, \epsilon_t^2)$$

where $x_t$ is the sample at time $t$, $\nabla_x \log p(x_t)$ is the gradient of the log-likelihood of the target distribution, $\alpha_t$ is the step size, and $\mathcal{N}(0, \epsilon_t^2)$ is the noise added to the sample.
![Unconditional Generation Visualization](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/visualizations/cfg_trajectory_curvature_0.gif?raw=true)
*Figure 1: Unconditional generation from a Flow Matching model*

While diffusion models learn to predict the score function, i.e. one component of $\nabla_x \log p(x_t)$, flow matching learns to predict the velocity field itself, i.e. $v_t = x_{t+1} - x_{t}$. The diffusion models are learning to denoise the noisy sample, while flow matching learns the linear interpolation between the current sample and the next one. 

For simplicity, we will assume a flow matching model for the rest of the post.

![Diffusion/Flow Model Visualization](/images/diffusion_flow_visual.svg)

*Figure 2: (a) A single forward pass through the model takes noisy input x_t and timestep t, producing a velocity/score prediction. (b) Iterative sampling follows these velocity predictions step-by-step from noise distribution to data distribution. (c) The model learns a vector field that transports samples from a Gaussian distribution $\mathcal{N}(0, 1)$ to the target distribution $p(\mathbf{x})$*

## Classifier Guided Generation
A natural design choice is to train the generative model on the entire dataset unconditionally and then use a classifier to guide the sampling process. However, this approach has two major drawbacks: 
1. The mode-collapse: an unconditionally trained model may not be able to capture the low density regions, areas with fewer training samples, of the target distribution well. 
2. Gradient issues: guidance takes the form of: 

$$\nabla_x \log p(x_t, c) = \nabla_x \log p_{g}(x_t) + \lambda \nabla_x \log p_{cls}(c \mid x_t)$$

where $p_{g}(x_{t})$ is the unconditional distribution, $\lambda$ is a hyperparameter that controls the strength of the guidance and $p_{cls}(c \mid x_t)$  is the conditional distribution coming from a classifier.  

First, the classifier should have learned a good representation of the condition such that the gradient of the conditional distribution is usable for the sampling process. Second, the difficulty lies combining the gradients effectively by tuning $\lambda$ during inference [^dhariwal2021diffusion].

## Conditional Generation
So far, we have only seen the unconditional generation case. Conditioning is the process of guiding the sampling process towards a specific region of the target distribution and can be fed to the model via class label, text prompt, or other information. 

![Conditional Generation Visualization](/images/classifier_free_guidance_visual.svg)

*Figure 3: (a) Single forward pass with conditional information c added to the model inputs. The velocity output now depends on both the noisy sample x_t, timestep t, and condition c. (b) Conditional sampling where c = left eye guides the trajectory specifically toward the left eye region of the target distribution.*

[^dhariwal2021diffusion] also noted that the classifier-guided generation can still be helpful for an conditionally trained model, even outperforming without guidance as well as unconditionally trained models with classifier guidance. The sampling now utilizes: 
$$\nabla_x \log p_{g}(x_t \mid c) + \lambda \nabla_x \log p_{cls}(c \mid x_t)$$

Though the two terms look similar, it is not well understood why this is exactly beneficial. But the difficulty still remains in combining the gradients effectively by tuning $\lambda$ during inference. This is where classifier-free guidance comes in.


---
    WIP 
---

# Classifier-Free Guidance

The key insight is to use make the model learn both the conditional and unconditional distributions during training. A neat Bayesian trick proposed in [^ho2021classifierfree], allows to then make the sampling iterations to boost towards the condition:

$$x = x_{uncond} + \gamma \cdot (x_{cond} - x_{uncond})$$

where $x_{uncond}$ is the generated sample without the class label, and $x_{cond}$ is the generated sample with the class label.
The guidance scale $\gamma$ is a hyperparameter that controls the strength of the guidance. A higher guidance scale will produce more class-specific samples.

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

# Visualizations

![CFG Visualization](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/non_overlapping/visualizations/both_classes_cfg.png?raw=true)
*Figure 4: Non overlapping classes: CFG comparison for both classes side by side.*

![CFG Visualization](https://github.com/VigneshSrinivasan10/flow-visualizer/blob/classifier-free-guidance/outputs/flow_gaussians/overlapping/visualizations/both_classes_cfg.png?raw=true)
*Figure 5: Overlapping classes: CFG comparison for both classes side by side.*

## Temperature Tuning

Is classifier-free guidance a form of temperature tuning? The answer is yes and no.

Yes, because the guidance scale $\text{cfg_scale}$ > $1.0$ can be interpreted as a way to reduce the temperature of the sampling process.
No, because the outcome is a condensation of the distribution towards the class label. It is not a simple temperature reduction. Decreasing the $\text{cfg_scale}$ will not always result in a sharper distribution. In fact, it will result in a more blurred distribution.

## References

[^dhariwal2021diffusion]: Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." In *Advances in Neural Information Processing Systems* 34 (2021): 8780-8794.
[^ho2021classifierfree]: Ho, J. and Salimans, T. "Classifier-Free Diffusion Guidance." In *Advances in Neural Information Processing Systems* (pp. 1-13). 2021.