---
layout: post
title: "On Expectation Maximization and Gaussian Mixture Models"
date: 2025-10-08
tags: [rust, machine-learning, backend, learning] ---
---

**Implementations**
- [KMeans](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/kmeans.rs)
- [Gaussian Mixture](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/gaussian_mixture.rs)

### A First Glance at Statistical Performance

![Model Performance](./assets/kmeans_gmm_clusters.png)

Today marks an exciting milestone! I want to cover one of the final implementations in my exploration of building all core Statistical Learning techniques from scratch.

**Checklist of completed models:**

- ~~Artificial Neural Network~~
- ~~K Nearest Neighbors~~
- ~~Singular Value Decomposition & Randomized SVD~~
- ~~Decision Tree, Random Forest & GBM~~

**New model type unlocked:**  
_Expectation Maximization (EM)_

The graph above shows the Gaussian Mixture Model's fit performance with KMeans initialization. In this article, I’ll go through implementing Expectation Maximization from scratch, and discuss some statistical and programmatic considerations.

To reproduce the example:

```bash
git clone https://github.com/cyancirrus/stellar-math
cargo run --example gmm
open kmeans_gmm_clusters.png
```


### Introduction

We’ll first cover the primitive types used in this implementation: KMeans and Gaussian Mixtures. Both are closely related, though conceptually distinct.

**Gaussian Distribution Recap:**
The Gaussian distribution, also known as the Normal distribution, is ubiquitous in statistics and named after mathematician Carl Friedrich Gauss. Formally:

$\text{Gaussian} = \text{Normal} :: N(\mu, \sigma^2)$

**KMeans vs GMM:**

* **KMeans:** Fits (k) Gaussian distributions with identical variance. Each data point receives a *hard* assignment.
* **GMM:** Fits (k) Gaussian distributions with potentially different variances. Each data point receives a *soft* assignment.

Both methods are widely used to cluster data into distinct segments.

### Preliminaries — Fitting KMeans

**Known properties of KMeans:**

* Equal variance
* Deterministic assignments
* Mixture parameters (centroids)

**Algorithm steps:**

1. Randomly initialize cluster means and assume equal probability for each centroid.
2. Assign each point to the closest cluster based on Euclidean distance.
3. Update cluster means.
4. Iterate until convergence (when mean updates are small).

**Gaussian distance simplification:**

$ d^*(x; \mu) = (x - \mu)'(x - \mu)$

This is equivalent to the squared Euclidean distance to the cluster centroid. The computational cost of KMeans:

$O(n \cdot d \cdot k)$

An optimization trick:

$(x-\mu)'(x-\mu) = \langle x,x \rangle + \langle \mu,\mu \rangle - 2 \langle x,\mu \rangle$

### Expectation Maximization for GMMs

GMMs introduce probabilistic assignments. The probability that a point belongs to cluster (k) is called the *mixing parameter*:

$pi_k(x) = \frac{\gamma_k(x)}{\sum_j \gamma_j(x)}$

The total probability of a data point:

$P(x) = \sum_k \pi_k(x) f_k(x; \mu_k, \Sigma_k)$

where (f_k) is the Gaussian PDF with potentially distinct covariance (\Sigma_k).

**EM steps:**

1. **Expectation (E-step):**

   * Use current parameters to compute cluster probabilities for each data point.
   * Normalize probabilities so they sum to 1.

2. **Maximization (M-step):**

   * Update (\mu_k) and (\Sigma_k) for each cluster based on weighted averages.
   * Rescale mixing parameters.

Iterate until convergence.

### Statistical Considerations

**Probability scaling and floating-point stability:**
Naively computing probabilities can lead to underflows. Using log probabilities helps:

$ \ln f(x; \mu, \Sigma) = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln\vert \Sigma \vert - \frac{1}{2} (x-\mu)'\Sigma^{-1}(x-\mu) $

Rescale using the maximum log probability to prevent negative infinity:

$l^*(x) = l(x) - \max_k l(x;\text{cluster}_k)$

```rust
let mut max_ln_prob = f32::MIN;
for k in 0..self.centroids {
    for c in 0..self.cardinality {
        let val = x_i[c] - self.means[k][c];
        x_bar[c] = val;
        z_buf[c] = val;
    }
    probs[k] = self.mixtures[k].ln() + ln_gaussian(&mut x_bar, &mut z_buf, dets[k], &lus[k]);
    max_ln_prob  = max_ln_prob.max(probs[k]);
}
let mut scaler = EPSILON;
for k in 0..self.centroids {
    probs[k] = (probs[k] - max_ln_prob).exp().max(EPSILON);
    scaler += probs[k];
}
```

**Covariance matrix computation:**
Use LU or Cholesky decomposition to solve $(x-\mu)'\Sigma^{-1}(x-\mu)$ efficiently.

```rust
fn ln_gaussian(x_bar:&mut Vec<f32>, z_buf:&mut Vec<f32>, det:f32, lu:&LuDecomposition) -> f32 {
    debug_assert_eq!(x_bar.to_vec(), z_buf.to_vec());
    let card = x_bar.len();
    lu.solve_inplace_vec(z_buf);
    let scaling = dot_product(&z_buf, &x_bar) / 2_f32;
    -(card as f32 / 2f32) * (2f32 * std::f32::consts::PI).ln()
    - 0.5f32 * det.ln()
    - scaling
}
```

### Implementation Overview

EM can be implemented efficiently by computing soft assignments and immediately updating centroids:

```rust
pub fn expectation_maximization(&mut self, data:&[Vec<f32>]) {
    let mut sum_linear = vec![vec![0_f32; self.cardinality]; self.centroids];
    let mut sum_squares = vec![generate_zero_matrix(self.cardinality, self.cardinality); self.centroids];
    ...
}
```

### Recap and Takeaways

* EM elegantly combines statistical reasoning with numerical programming tricks such as triangular solves, computational reuse, and log stabilization.
* Initializing GMMs with KMeans centroids helps convergence.
* With this, the basic Statistical Learning algorithms are fully implemented in Rust from scratch!

I think my next steps will be to explore some Control Theory in C++ or perhaps extending Fourier transforms to arbitrary dimensions.

Keep coding!
