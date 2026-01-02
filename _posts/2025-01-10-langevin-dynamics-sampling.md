---
layout: post
title: "Langevin Dynamics for Sampling"
date: 2025-01-10
categories: [machine-learning, sampling]
tags: [langevin, mcmc, sampling, stochastic-processes]
math: true
excerpt: "How Langevin dynamics can be used to sample from complex distributions, and the relationship between step size and effective temperature."
---

Langevin dynamics provides a powerful framework for sampling from probability distributions. Let's explore the mathematics and practical considerations.

## The Langevin Equation

The overdamped Langevin dynamics is given by:

$$dx = \nabla \log p(x) \, dt + \sqrt{2} \, dW_t$$

In discrete form with step size $\eta$:

$$x_{t+1} = x_t + \eta \nabla \log p(x_t) + \sqrt{2\eta} \, \epsilon_t$$

where $\epsilon_t \sim \mathcal{N}(0, I)$.

## Temperature and Step Size

An important insight: the effective temperature of Langevin dynamics is:

$$T_{\text{eff}} = \frac{\sigma^2}{2 \cdot dt}$$

where $\sigma$ is the noise scale. This relationship is crucial for understanding convergence and mixing.

## Implementation

```python
def langevin_step(x, score_fn, step_size, noise_scale=None):
    """Single step of Langevin dynamics."""
    if noise_scale is None:
        noise_scale = np.sqrt(2 * step_size)
    
    score = score_fn(x)
    noise = np.random.randn(*x.shape)
    
    return x + step_size * score + noise_scale * noise
```

## Annealed Langevin

For better sampling, we can anneal the temperature:

$$\sigma_t = \sigma_{\max} \cdot \left(\frac{\sigma_{\min}}{\sigma_{\max}}\right)^{t/T}$$

This helps escape local modes early and refine samples later.

---

*Next: Combining Langevin dynamics with learned score functions.*
