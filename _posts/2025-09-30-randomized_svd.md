```
layout: post
title: "Computational Implementation of Randomized SVD"
date: 2025-09-18
tags: [rust, machine-learning, backend, learning]
```

[Randomized SVD Algorithm](https://github.com/cyancirrus/stellar-math/blob/main/src/solver/randomized_svd.rs)  
[Golub-Kahan Implementation](https://github.com/cyancirrus/stellar-math/blob/main/src/decomposition/svd.rs)  
[Optimized QR Decomposition](https://github.com/cyancirrus/stellar-math/blob/main/src/decomposition/qr.rs)  
[Givens Implementation](https://github.com/cyancirrus/stellar-math/blob/main/src/decomposition/givens.rs)  

[Algorithm Source :: N. Halko, P. G. Martinsson, A. Tropp](https://arxiv.org/pdf/0909.4061)

## Introduction

Singular Value Decomposition (SVD) is one of the cornerstones of numerical, scientific, and statistical computing. It underpins everything from PCA to large-scale linear algebra pipelines, enabling efficient updates and approximations without exploding computational costs.

In this post, we’ll explore SVD from both a **computational** and **statistical** perspective. Specifically, we’ll cover:

- Optimizing matrix operations for performance.  
- Leveraging statistics for lower-rank approximations.  

We’ll implement ideas from Halko, Martinsson, and Tropp’s seminal work *Finding Structure in Randomness*, and highlight how careful computational tricks make SVD practical at scale.


## What is SVD?

At a high level, SVD transforms a matrix into three components:

1. **Input rotation (U)**  
2. **Scaling (Σ)**  
3. **Output rotation (Vᵀ)**

Intuitively, SVD identifies the “directions” in your data that capture the most signal.  

For example, suppose we’re modeling the likelihood of patients revisiting a hospital, given demographic and behavioral variables:

```md
- age
- gender
- region
- prescription adherence
- preventive care
- previous-year activity
- BMI
- …plus 20+ other variables
```

The raw input space is large and complex. But often, a few combinations of variables dominate the main signal. Perhaps “age × preventive care × prescription adherence” explains most of the variance. SVD compresses this multivariate data into latent features: the first singular vector captures the strongest pattern, the next captures the next strongest, and so on.

By truncating to the top `K` components, we remove noise while retaining most of the meaningful signal. This is the foundation of PCA, low-rank approximations, and efficient numerical computation in ML pipelines.



## Implementing SVD

### Deterministic SVD

The classic algorithm involves two steps:

1. **Bidiagonalization**

   * Golub-Kahan procedure using Householder reflections.
   * Columns below the diagonal and rows beyond the first superdiagonal are zeroed.

2. **Bulge chasing**

   * Givens rotations and extensions diagonalize the bidiagonal matrix.
   * This produces the singular values and vectors.

> Note: Direct diagonalization without bidiagonalization generally only converges for symmetric positive-definite matrices.



### Randomized SVD

Randomized SVD accelerates computations for large matrices by approximating the subspace spanned by the top singular vectors. The procedure is:

```text
1. Generate a random matrix Ω ~ (m × k)
2. Form Y = A * Ω
3. Orthonormalize Y via QR decomposition → Q
4. Project A into the smaller subspace: B = Qᵀ * A
5. Compute deterministic SVD on B
6. Recover approximate U, Σ, V from the small SVD
```

This reduces computational cost while capturing the dominant signal, which is often all you need in practical applications.



## QR Decomposition: The Core Building Block

QR decomposition is central to both randomized SVD and Golub-Kahan. Given a matrix `A`, we find:

```
A = Q * R
```

Where `Q` is orthogonal and `R` is upper-triangular. Key ideas:

* Householder reflections rotate vectors so that elements below the diagonal become zero.
* Reflections are rank-one symmetric matrices: `Q[i] = I - β * u * uᵀ`.
* By chaining reflections, we compute `Q = Q[1] * Q[2] * ... * Q[n]`.

#### Computational Optimization

Naively, applying `Q` to another matrix costs `O(n^4)` for dense matrices. But by exploiting the structure of Householder reflections, we can reduce this to `O(n^3)`:

```rust
// Householder-based QR update
let w = beta * Q_km1 * v_k;
A -= w * v_k';
```

This optimization is crucial in practice, especially for large-scale pipelines and in Golub-Kahan bidiagonalization.



### QR → Golub-Kahan

Golub-Kahan extends QR ideas to bidiagonalization:

* Columns below the diagonal are zeroed (like QR).
* Rows beyond the superdiagonal are zeroed.
* Iterating column and row zeroing produces the bidiagonal matrix.

The same optimizations used in QR reduce the cost of these operations, making the algorithm tractable for large matrices.



## Statistical Optimization: The Randomized Payoff

Suppose we have two `10,000 × 10,000` matrices. Naively, computing `A * B` requires `O(n^3) ≈ 10^12` FLOPS—a trillion operations!

With randomized SVD, if we approximate with rank `k = 1,000`:

```
Flops ≈ O(n^2 * k) = O(10^8 * 1e3) = 10^11
```

We do only ~10% of the work, while preserving the dominant signal. This dramatically accelerates pipelines in ML and scientific computing, compounding over repeated matrix operations.


## Wrapping Up

SVD is a powerful tool bridging linear algebra, statistics, and computational optimization. Through careful QR and Golub-Kahan implementations, we can handle large matrices efficiently. Randomized SVD then provides a statistical shortcut, allowing us to approximate dominant structures with far less computation.

The combination of algebraic tricks, computational insights, and statistical reasoning makes this one of the most elegant examples of applied numerical linear algebra.

Explore the [codebase](https://github.com/cyancirrus/stellar-math) to see these ideas in action—and watch the FLOPS disappear.

Thanks for reading!
