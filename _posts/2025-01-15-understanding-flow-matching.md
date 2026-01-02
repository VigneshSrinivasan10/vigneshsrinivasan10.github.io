---
layout: post
title: "Understanding Flow Matching for Generative Models"
date: 2025-01-15
categories: [machine-learning, generative-models]
tags: [flow-matching, diffusion, deep-learning]
math: true
excerpt: "A deep dive into flow matching, a simpler alternative to diffusion models for learning continuous normalizing flows."
---

Flow matching has emerged as an elegant framework for training generative models. Unlike diffusion models that require complex noise schedules and score matching objectives, flow matching provides a more direct path to learning probability flows.

## The Core Idea

The key insight is to define a *conditional* probability path from noise to data:

$$p_t(x | x_1) = \mathcal{N}(x; \mu_t(x_1), \sigma_t^2 I)$$

where $x_1$ is a data sample, and we interpolate from pure noise at $t=0$ to the data point at $t=1$.

## The Simplest Interpolation

For optimal transport, we use linear interpolation:

$$\mu_t(x_1) = t \cdot x_1$$
$$\sigma_t = 1 - t$$

This gives us the conditional flow:

$$x_t = t \cdot x_1 + (1-t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

## Training Objective

The velocity field we need to learn is simply:

$$u_t(x_t | x_1) = x_1 - \epsilon$$

So our loss becomes:

$$\mathcal{L} = \mathbb{E}_{t, x_1, \epsilon} \left[ \| v_\theta(x_t, t) - (x_1 - \epsilon) \|^2 \right]$$

## Code Example

Here's how you might implement the training step:

```python
def flow_matching_loss(model, x1, t=None):
    """Compute flow matching loss."""
    batch_size = x1.shape[0]
    
    # Sample time uniformly
    if t is None:
        t = torch.rand(batch_size, 1)
    
    # Sample noise
    eps = torch.randn_like(x1)
    
    # Interpolate
    x_t = t * x1 + (1 - t) * eps
    
    # Target velocity
    target = x1 - eps
    
    # Predict and compute loss
    pred = model(x_t, t)
    return F.mse_loss(pred, target)
```

## Why This Works

The magic is that when we marginalize over $x_1$, the conditional flows "average out" to give us the marginal probability path from the noise distribution to the data distribution.

## Next Steps

In the next post, we'll explore how to combine flow matching with classifier-free guidance for conditional generation.

---

*Have questions or comments? Feel free to reach out!*
