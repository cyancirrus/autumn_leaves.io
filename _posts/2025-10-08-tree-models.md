---
layout: post
title: "Implementation of Tree based Models"
date: 2025-10-08
tags: [rust, machine-learning, backend, learning]
---

[Gradient Boost Implementation](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/gradient_boost.rs)  
[Random Forest Implementation](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/random_forest.rs)  
[Decision Tree Implementation](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/decision_tree.rs)  

## Introduction

Machine learning is based upon a few based structures and methods - the perceptron, the decision tree, expectation maximization, clustering, dimension reduction.
Today we'll be going through how to implement the following ML Algorithms:
 
 - Gradient Boosting
 - Random Forests
 - Decision Tree

For these algorithms this article will be looking at their algorithmic definition, their complexity and also look at their statistical performance on a real dataset.

## Preliminaries - What is a Decision Tree?

Lets consider the lowly if-else statement, this way we can build intuition for what is a decision tree. Consider the question on whether I should bring an umbrella with my outside...

```rust
if it_is_raining  { // then
    items_to_carry += "umbrella";
} else { // it is not raining
    // do not need to carry umbrella
}
```

As we can see the above if-else statement will intelligently help us to decide whether what we should do if it is raining outside.
Lets now extend this if-else statement into the data context, consider the problem of trying to model the quality of a wine.

Consider the factors that would be helpful with predicting quality
 - country of origin // france, portugal, italy are all known for their wines
 - year of production
 - average rainfall per country for the year
 - average temperature per country for the year
 - brand who has produced the wine
 - colour - white / red / red-blend 
 - type of wine which has been produced // merlot, pintot
 - price
 - Quality <- This will be what we are trying to predict

All of these would be very useful and lets imagine the following if condition
```rust
if country == greece & colour == green {
    // Greece has amazing white and green wines
    predicted_quality = 1000_f32;
}
```

The above simply is the backbone for how decision trees work to predict results. Imagine hundreds of branches which could cover different conditions for all countries and would split the data into different if-then statements intelligently - this is the decision tree.

Algorithmically the decision tree performs the following
```rust
let mut best_split = None
let mut explanatory_power =  0;
for _ in 0..desired_number_of_nodes {
    for each dimension find best split {
       if explanatory_power(this_split) == best {
          best_split = this;
          explanatory_power = explanatory_power(this_split);
       }
    }
}
```

All the above does is formalize the if/else condition, and the predicted `Quality` for each node is simply the average quality of the items within the node. Remember to visualize as we are splitting data, the number of items in each node increasingly decreases, so we aren't reclassifying the entire data at each step, only the members which are assigned to the current node.

```md
Ie Node[0] // contains all data
- split and now we no longer within training consider Node[0] as it is not a leaf node
Node[1], // contains ~ 1/2 data
Node[2], // contains ~ 1/2 data
- split Node[2]
Node[1], // contains ~ 1/2 data
Node[3], // contains ~ 1/4 data (was previously node 2)
Node[4], // contains ~ 1/4 data (was previously node 2) 
```




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
2. Form Y = (A A') A * Ω
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
