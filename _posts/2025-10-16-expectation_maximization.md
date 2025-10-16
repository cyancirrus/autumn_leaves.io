---
layout: post
title: "On Expectation Maximization and Gaussian Mixture Models"
date: 2025-10-08
tags: [rust, machine-learning, backend, learning]
---

**Implementations**
- [KMeans](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/kmeans.rs)
- [Gaussian Mixture](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/gaussian_mixture.rs)

### A First Glance of the Statistical Performance

![Model Performance](./assets/kmeans_gmm_clusters.png)

Today, marks an exciting bookarmark! Today I wish to cover one of the last implementations for my explorations for implementing all Statistical Learning main Techniques from Scratch!

Checklist
- ~Artificial Neural Network~
- ~K Nearest Neighbors~
- ~Singular Value Decomposition & Randomized Svd~
- ~Decision Tree & Random Forest & GBM~

* New model type unlocked!
_ Expectation Maximization_

The graph above shows the Gaussian Mixture Model's fit performance with Kmeans initialization.
In this article today I'm going to be going through how to implement Expectation Maximization (EM) from scratch, and a couple of Statistical and Programatic techniques.

To reproduce:
```bash
git clone https://github.com/cyancirrus/stellar-math
cargo run --example gmm
open kmeans_gmm_clusters.png
```

### Introduction

Lets cover the primitive types for this implementaiton, the first being KMeans, and the Second Gaussian Mixtures. Both are extremely similar and heavily related and we partition the two into two separate camps.

First lets have a recall what is the Gaussian Distribution. The gaussian distribution is another name for the Normal Distribution, just references famous mathematician Frederick Gauss instead of calling it the "normal" distribution. Gaussian is the same normal everyone in statistics have grown up using and is used in the CLT, ie `Gaussian == Normal`.

KMeans := k different Gaussian (Normal Distribution) functions with identical variance are fit to the data. Each datum is given hard assignment.
GMM := k different Gaussians with non-identical variances are fit to the data. Each datum is given soft assignment.

Both KMeans and GMM are used when we wish to cluster data into different categories or segmentations of like others.


### Preliminaries - How should we look at fitting KMeans the simpler case?

So lets start with our knowns

- kmeans has mixture paramters
- kmeans has equal variance
- kmeans has deterministic

The first step for kmeans will be to randomly initialize our cluster means' and presume equal probability for each centroid.
Next lets take a look at the gaussian distribution.

$gaussian(x; mu, sigma) = 1/(root pi)^2 sigma * exp( -1/2sigma^2(x-u)'(x-u))$

As we know from statistics and mathematics that maximizing f(x) and maximizing g(f(x)) when  g is monotonic are exactly identical.

Looking at the 
$ln f(x; mu, sigma) = - 1/2 ln (root pi sigma^2) - 1/2 sigma^2(x-mu)'(x-mu)$

Remembering now that the variance parameter is defined to be equal and that the front term is now constant we can simply look at

$ d{star}(x;mu) = (x - mu)'(x - mu)$

Just a quick note here, since we reversed the sign underneath simplification we'll now be looking at taking the minimum of $d{star}$ for the best distribution. Convienantly this is exactly equal to the distance metric from a clusters centroid, so simply _we are finding the cluster which is closest to our data point_ this will be our cluster assignment.

Now simply we will iterate over the data, assigning the points to the closest cluster, and then updating the cluster means until convergence (ie the update for the means is quite small).

Perfect! Elegant, simple.

The training cost for Kmeans
$O(d \cdot n \cdot d)$

One could use the following in order to help optimize the kmeans order but asymptotically runtime will still be proportional to $O(alpha \cdot N)$.
$(x-mu)'(x-mu) = \<x,x\> + \<mu,mu\> -2\<x,u\>$

### Expectation Maximization - How to obtain convergence in correlated state for GMM's

The main issue for Gaussian Mixture models is that we have the following for the probability

Probability that the data applies to the kth clusters, $PI_k$ are referred to as the mixing parameters.
$PI_k(x) = GAMMA_k(x) / Sum GAMMA_k(x)$

Probability of x is the product of the probability it applies to kth cluster multiplied by the probability of said datapoint for the cluster.
$Pr(x) = Sum PI_k(x) * f_k(x; MU, SIGMA)$

Remember the $f_k(x; MU, SIGMA)$ simply refers to gaussian distribution, however most importantly with different variances.

As we can see here, there are two main parts $PI_k(x)$ and $f_k(x; MU, SIGMA)$.

The expectation maximization procedure proceeds as follows.

* Expectation *

- for the current iteration utilize the previous mixing parameters
- for each datapoint determine the cluster probability
- normalize the probabilities for the datum to be equal to one


* Maximization *

- update the paramters $MU$ and $SIGMA$ for each cluster dependent upon this new information
- finally rescale the mixing parameters


We simply loop over this structure until we obtain convergence.

### Statistical Considerations

#### Probability scaling and floating point numbers 

There exists a bit of a problem area in the naive implementation of GMM, the main point being that some probabilities will rapidly converge to 0. This will then present division by zero, and we'll return NaN's and it severly hurts convergence.

Instead of using the raw probabilities we can utilize a similar trick to kmeans by considering the log probabilities.

However, as variances are different we must be a bit more careful
$gaussian(x; mu, sigma) = 1/((2 \cdot PI)^d/2 |SIGMA|) * exp( -1/2(x-MU)' SIGMA^-1 (x-MU))$

Logging the above distribution we obtain
$l(x; mu, sigma) = d/2 * ln( 2 * PI) - 1/2 ln(|SIGMA|) -1/2(x-MU)' SIGMA^-1 (x-MU))$

Setting asside temporarily the problem with the inversion of the Covariance Matrix $SIGMA$. Lets first concentrate on the relations we know.
$PI_k(x) = GAMMA_k(x) / Sum GAMMA_k(x)$

Lets consider what would happen if we were to multiply all probabilities by a constant
$PI_k(x) = \cdot GAMMA_k(x) / Sum ALPHA \cdot GAMMA_k(x)$

Pulling this to the front we can easily see we obtain the same result.
Consider if we were to divide by the maximum of the probabilities then

$l{star}(x; MU, SIGMA) = l(x; mu, sigma) - max l(x; cluster[k])$


This then rescales the probabilities so that we can avoid the underflows of negative infinities which become NaN's. We simply rescale and by using
$ exp( ln(a) - ln(b)) = exp(ln(a/b) = a/b$

We are simply at a much better point for the implementation

<details>
<summary>Click to expand code</summary>
{% highlight rust %}
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
{% endhighlight rust %}
</details>

#### Solving for the Complexities of the Covariance Matrix

Simply use LU or Cholesky decomposition. I've personally used LU decomposition because I had already written the methods but lets take a look at our statements (but note Cholesky would be quicker due to the symmetry of the Covariance Matrix and the same assumption in Cholesky).

Lets consider the exponential part of the probability for the gaussian distribution, specifically the core part ie
$(x-MU)' SIGMA^-1 (x-mu)$.

lets first centralize x so we don't have so many floating symbols
$x{bar}' SIGMA^-1 x{bar}$.

Lets consider solving just a part of the above ie

$SIGMA^-1 x{bar} = z$
$ x{bar} = SIGMA z$

Now we can just use LU backwards solve which showing quickly
$LU = SIGMA$
$x{bar} = LU z$
$x{bar} = Lz{star}$

This infers $z{star}$ which then we have the equation.
$Uz = z{star}$ which infers $z$. Next we can finally plug into our original exponential centralization
$x{bar}' SIGMA^-1 x{bar} = $x{bar}'z$

And we're nearly finished. Lastly we need the determinant of Sigma.
Recalling that LU are both traingle this then infers
$|SIGMA| = |LU| = |L||U| = |U| = Product u_ii$

Recalling that L diagonals are simply 1. So we simply sum over the diagonal!


<details>
<summary>Click to expand code</summary>
{% highlight rust %}
fn ln_gaussian(x_bar:&mut Vec<f32>, z_buf:&mut Vec<f32>, det:f32, lu:&LuDecomposition) -> f32 {
    // xbar := x - mean;
    // we have x'Vx, where V := 1/ self.variance
    // solve sub problem LUx = z*; for z* and then <x, z*>
    debug_assert_eq!(x_bar.to_vec(), z_buf.to_vec());
    let card = x_bar.len();
    lu.solve_inplace_vec(z_buf);
    let scaling = dot_product(&z_buf, &x_bar) / 2_f32;
    {
        -(card as f32 / 2f32) * (2f32 * std::f32::consts::PI).ln()
        - 0.5f32 * det.ln()
        - scaling
    }
}
{% endhighlight rust %}
</details>

### Computational Considerations - A logical merger of EM steps

Perfect in this section I just wish to note the structural form of EM when written into code.

<details>
<summary>Click to expand code</summary>
{% highlight rust %}
pub fn expectation_maximization(&mut self, data:&[Vec<f32>]) {
    let mut sum_linear = vec![vec![0_f32; self.cardinality]; self.centroids];
    let mut sum_squares = vec![generate_zero_matrix(self.cardinality, self.cardinality); self.centroids];
    ... 

    for x_i in data {
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
        for k in 0..self.centroids {
            let pr = probs[k] / scaler;
            nweighted[k] += pr;
            for c in 0..self.cardinality {
                sum_linear[k][c] += pr * x_i[c];
            }
            for i in 0..self.cardinality {
                for j in 0..=i {
                    sum_squares[k].data[i * self.cardinality + j] += pr * x_i[i] * x_i[j];
                }
            }
        }
        ...
    }
}
{% endhighlight rust %}
</details>

Quickly notice instead of performing the entire expectation step where we are determining the soft-assignments, we merely find the soft assigments and then immediately update the new centroids. While this does require some messing about in combining the `sum_linear` and the `sum_squares` terms and scaling it's really quite straightforward. Just requires some clear naming conventions.


### Recaps and Takeaways

Thanks for reading! I'm extremely excited at having finally conquered the basic Statistical Learning Algorithms within rust. I've hoped you have learned something along the way with me as well.
Expectation Maximization is a wonderful technique and combines disparate areas of other numerical programming such using triangular solving, computational reuse, and using logs for stabilization.
Oh -- also utilize kmeans for the initial estimization for the centroids, it seems to get near monotonically better performance and can help with convergence.

I think next I'll either tackle some Control within C++ or look at extending my Fourier transforms to handle arbitrary dimensions.
Keep coding!
