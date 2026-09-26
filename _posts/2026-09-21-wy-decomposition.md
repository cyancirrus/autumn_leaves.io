---
layout: post
title: "WY Decomposition"
date: 2026-9-21
tags: [rust, planning, perception, action, learning]
---

**Implementations**
- [Main Repository](https://github.com/cyancirrus/stellar_math)
- [Wy Decomposition](https://github.com/cyancirrus/stellar-math/tree/main/src/decomposition/wy)
- [Benchmark Script](https://github.com/cyancirrus/stellar-math/blob/main/scripts/lq_decomposition.sh)

# WORK IN PROGRESS ARTICLE
## TODO: Finish article and then split out into the sections into articles and functionally add article type filters

## QR Decomposition

_highlighted terms should be able to be found in article glossary_

A `matrix` is a system of numbers defined by having `observations X features ` where a feature is a type of data, eg height, velocity, frequency, colour.
Matrices are used universally from medicine, to game design, to image editing software, to machine learning and engineering applications.

An Example of a Matrix ie the Identity Matrix:

$$
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

However, many of the formulas within these fields are strictly impractical if not impossible to compute if we naively implement them the way the math says to do so.
Because computers when ultimately zoomed down compute in terms of bits and bytes, we have finite precision - and more importantly error itself will compound over matrix multiplication not simply addition - so no matter the finite precision which we would use, we are in still need of numerical methods - if we care at all for how long our method takes to calculate, or if we have any budget for memory whatsoever.

`QR` Decomposition is a specific way to split a `matrix` into component parts, such that when we multiply them we obtain the same values, and in reality with a very small amount of numerical error.
ie

$$
A = Q R + \epsilon
$$

However, this is not like in statistics where we have a statistical error term due to the compression of the system ie that the number of features.
Remembering $\hat{Y}$ is _almost never_ near-exact unlike QR's $\epsilon$, numerical noise sitting near the machine precision limit ie $1e-7$ (float 32).

$$
\hat{Y} = X B + \sigma e
$$

In reality is almost better for the sake of this paper to presume that QR is exact or near-exact ie $$A = QR$$. 

Given the `QR` Decomposition we can now effectively solve systems of equations as well as drmatically reduce the amount of work as well as the error.
The computer does not have infinite precision so it really matters how we go about manipulating numbers.

QR as a decomposition is just super handy as a way to solve a system of equations quickly, and can be used to drastically simplify problems, or to change their computation properties.
QR is a non-iterative algorithm so it runs in detiministic time as compared to SVD and it's constant factors for inference are near on par with SVD.

QR is one of the major tool for any industry which uses mathematics as a vehicle for insights or information, signal processing, downscaling for randomized SVD.

## WY Decomposition - Main Focus and Post Overview

In the coming sections, We will first hit the glossary as a reference point for definitions, then we'll begin the tour of row major, simd and kernels, wy and then finally computational optimizations
**Goals**
- Provide intuition for why the LQ decomposition is used in row major
- Explore the derivation for WY and show a more optimal representation of T for row major form

Promise this will be worth it here are my results against Rust's most respected numerical library
**TODO: change this link to of my benchmark results for small matrices**
![Benchmark Results WY Decomposition](./assets/wy_benchmark_results.png) 

## Base Primitives Needed for WY

Before diving into algorithms, lets get some definitions out of the way that way there can be a common reference point for the coming sections.

#### WY Primitives

**Glossary of Terms**

_Click any of the following to expand_

<details markdown="1">
<summary><strong>Matrix Primitives</strong></summary>
> **Matrix** := A "2dimensional" grid of numbers which have coherency in its organization ie `observations X features`  
> **Feature** := A type of measurement ie height, quantity colour, frequency etcetera  
> **Observation** := A unit of an analysis ie a we surveyed fifty people and collected data, the person would be an observation  
> **Rows** := A row is a slice of data from a matrix which pertain to all observations of a particular as a feature  
> **Cols** := A col eg column, is a slice of data from a matrix which pertains to all features for an individual  
> **Transpose** := Take a matrix and flip it across the diagonal, ie interpretation of rows becomes cols and cols becomes rows $$\forall i,j \epsilon A_{ij} <- A_{ji}$$  
> **Decomposition** := The splitting of a matrix into different component parts such that we can reconstruct the matrix if needed ie $$A = QR$$  
</details>

<details markdown="1">
<summary><strong>Matrix Types Regarding Shape</strong></summary>
> **Square** := A matrix guaranteed as having the same number of rows as cols  
> **Rectangular** := A matrix not guaranteed to have the same number of rows as columns  
> **Unspecified** := A matrix is assumed to be rectangular  
> **Block** := A matrix of matrices, this is a logical way to "split" the matrix and merely is a clever way to refer to subportions of a matrix  
</details>

<details markdown="1">
<summary><strong>Matrix Types Regarding Numeric Density ie Where are the Zeros?</strong></summary>
> **Dense** := A matrix where there is a nonguarantee that any values are zeros  
> **Lower Triangular** := $$L$$ this is a square matrix where for where all elements above the diagonal are zero  
> **Lower Trapezoidal** := $$L$$ this is a rectangular matrix for where all elements above the diagonal are zero  
> **Upper Triangular** := $$R$$ this is a square matrix where for all elements below the diagonal are zero  
> **Upper Trapezoidal** := $$R$$ this is a rectangular matrix where for all elements below the diagonal are zero  
> **Diagonal** := a matrix having only entries on the diagonal
</details>

<details markdown="1">
<summary><strong>Matrix Types Regarding Properties</strong></summary>
> **Identity** := A Diagonal matrix containing only the value 1, $$A_{ij} = \delta_{i==j}$$;  
> **Orthonormal** := A matrix $$Q$$ s.t. $$Q Q' \equiv I$$ and $$Q' Q \equiv I$$, geometrically this means for $$A Q$$ we have taken the rows and rotated them and for $$Q A$$ the cols.
</details>

<details markdown="1">
<summary><strong>Matrix Abbreviations</strong></summary>
> **Q** := An orthonormal matrix  
> **L** := An left (lower) triangular matrix  
> **R** := A right (upper) triangular matrix  
> **QR** := A matrix decomposition which splits a matrix of numbers into component parts consisting of an orthonormal `Q` and a right (upper) triangular matrix `R`  
> **LQ** := The transposition of (QR)' ie `Right (Upper) Triangular` matrix becomes `Left (Lower) Triangular`  
> **WY** := A decomposition s.t. we replace `Q` with a more optimized hybrid (wrt time and memory) form `Q ~ (I - YTY')` ie not materialized 
</details>

<details markdown="1">
<summary><strong>Computation Technology</strong></summary>
> **Blas** := Basic Linear Algebra Subprogram (B.L.A.S.) foundational project for all of numerical and computational science started in 1970s still innovations today  
> **Fortran** := A programming language in which much of Blas written in 1970s, prior to then being ported to the programming language `C` the standards of which still define current archetecture  
> **Vector** := A vector is memory on the `heap` ie not `cache` where we allocate and pass around the `pointer` ie the `reference`  
> **Vector of Vectors** := A representation of matrix which appears like `vec![vec![row1], ..., vec![rowm]]` expensive because of indirection  
> **Row Major Form** := A linearized representation of matrix which appearing as `vec![a00, a01, ..., a0n, ..., am0, ... amn]` example below  
> **Col Major Form** := A linearized representation of matrix which appearing as `vec![a00, a10, ..., am0, ..., a0n, ... amn]` example below  
> **SIMD** := Same Instruction Multiple Data (S.I.M.D.) refers to advanced vector instructions which allow for speed of numerical computation via parallelization  
> **Kernel** := A specific computational program which processes subparts of `matrix computation` via `SIMD` and advanced vector instructions which allow for parallelization  
</details>

## Use cases for the QR decomposition

QR Decomposition is a way to decompose a matrix (a system of numbers) into a form in which we can then use numerically to solve for specific quantities and enables use.
The computer does not have infinite precision so it really matters how we go about manipulating numbers.

QR Decomposition specifically allows to solve for unknowns quickly and accurately once calculated

1) Solve $$Ax = y$$ for $$x$$ given knowledge of $$A$$ and $$y$$

<details markdown="1">
<summary>Click to expand Derivation</summary>
<div align="left">
$$
\begin{aligned}
A &\triangleq QR \\
A x &= y \\
&\implies \\
Q R x &= y \\
Q' (Q R x) &= Q' y \\
R x &= Q' y \\
w &\triangleq Q' y \\
R x &= w \\
&\implies \\
&x \text{ solved via back-substitution} \\
\end{aligned}
$$
</div>
</details>

QR Decomposition also allows us to project data for other algorithms and other learners dramatically reducing training time for certain algos

2) Dimension Reduction by working with the covariances

<details markdown="1">
<summary>Click to expand Derivation</summary>
<div align="left">
$$
\begin{aligned}
\text{Goal} &: f(X) \text{but fast} \\
X &\triangleq Q R \\
R &= Q' X \text{R available from QR}\\
&\implies \\
\exists g( R ) &: f(X) \approx Q * g( R ) \\
\end{aligned}
$$
</div>
</details>

For a specific instance of the solver consider we can look towards least squares and linear regression.

<details markdown="1">
<summary>Click to expand Derivation</summary>
<div align="left">
$$
\textstyle
\begin{aligned}
Y_t &= X_t B_t \\
&\implies \\
B_t &= (X_t' X_t)^{-1} X_t' Y_t \\
X_t &= QR \\
&\implies \\
B_t &= (R'Q'Q R)^{-1} R' Q' y_t \\
B_t &= (R' R)^{-1} R' Q' y_t \\
B_t &= R^{-1} R'^{-1} R' Q' y_t \\
&\text{recalling} \\
(C D)^{-1} &= D^{-1} C^{-1} \\
&\implies \\
B_t &= R^{-1} Q' Y_t \\
&\implies \\ 
(Q R) B_t &= Y_t \\
&\therefore \\
&B_t \text{ from using derivation (1) } \\
&\implies  \\
\hat{y}_{t+1} &= x_{t+1} B_t
\end{aligned}
$$
</div>
</details>

QR is one of the major tool for any industry which uses mathematics as a vehicle for insights or information, signal processing, downscaling for learning and efficiency.

## LQ Justification: The Want for a Row Major Representation of QR

### Row Major Form Definition and Clarification

`Row Major Form` (row-major) is a way to organize a `matrix` which instead of having a _vector of vectors_ instead we linearize the data by scanning `row` by `row`.
The problem with the `vector of vectors` approach for the representation is that we would have tons of pointer chasing and more specifically the prefetcher has to work so much harder that it becomes an inefficient way to represent the data.

Imagine the matrix:

$$
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 \\
2 & 3
\end{bmatrix}
$$

**Row Major form** we simply represent the above matrix as

```Rust
let (a00, a01, a10, a11) = (0f32, 1f32, 2f32, 3f32);
let rm = vec![a00, a01, a10, a11];

print!("rm: {rm:?}");
>> rm: [0, 1, 2, 3];
```

**Column Major Form** is a very similar way to organize data and would appear as
```Rust
let (a00, a01, a10, a11) = (0f32, 1f32, 2f32, 3f32);
let cm = vec![a00, a10, a01, a11];
print!("cm: {rm:?}");
>> rm: [0, 2, 1, 3];
```
These both refer to the _same_ matrix, they are just _different perspectives_ on how one would represent and subsequently process said data.
If you stare hard enough you can see that the `transpose` is the implicit connection between the two.
This traditionally has been the long standing interface between modern code which now represents data as `row major` and the `Blas` style kernels.

Essentially, historically *Fortran*, these mathematical concepts were all written in column major form.
However, nowadays, most apis and data comes in row-major form as that has been the main way that *Computer Science* (CS) has structured data and so there's been a natural barrier at which point every computation and decomposition would need to go through a layer of transposition prior to being calculated.

Even if the transposition can be `SIMD'd` this is still a full scan through the data which must happen 

However, analyzing this misalignment simply in terms of our storage format changing, is a bit less genuine than recognizing that the original form stored the data with `feature` as the contiguous major axis of iteration, and that our applications moved towards `observations` becoming the primary axis.

The following shows how a contiguous memory layout would appear, with each feature being an m-length vector, one for each unit of analysis ie the `observation` axis

$$
X_{feat} := \big[ feature_0 \mid feature_1 \mid \cdots \mid feature_n \big]
$$

All said and done, when solving exact systems, $A x = y$ for $x$, we can eliminate a significant amount of computation, by presuming $A$ is already within its required form with features within its rows.

I will explore this point further in an unrelated post, as this becomes genuinely complex.

### WY Derivation - the importance of the Row Major Form

The original QR is a masterpiece however it presumes a column major format.

One of the main benefits from the WY form is that we only need to have access to the current rows data for the householder vector in order to find it's zeros first consider the row-major representations of the following.

LQ, lets consider zeroing the $row_k$ we can simply consider the values, ie each of these are simple floats -> Order M.
```
[rk0, ..., rkn];
```


However let's look at QR when trying to zero the $Column_i$ in row major form we would need the following data
```
[row_0..k,r0k, row_0k+1],
[row_1..k,r1k, row_1k+1],
[      ...             ],
[row_m..k, r_mk, ...   ],
```
requiring nearly the entire matrix! -> Order M x N.

While the memory prefetcher is genius this genuinely thrashes the cache significantly if we represent the data in row major form.
This is why most libraries will transpose their data prior to using the QR decomposition so that it is in column major form.

If we continue at the LQ form of decomposition, and if we carry this detail further we can see that our triangle update becomes

### WY Mathematical Derivation and the Forced Lower Triangle T

I have not personally found a derivation for $T$ for the $WY(LQ)$ reprsentation, in any literature, so I thought I would provide it.

_Here I am going to presume some level of fluency in linear algebra while not being mathematically complete,
still is a meaningful gesture at the full derivation and will help provide mathematical intuition for the form of the full derivation_


<details markdown="1">
<summary><strong>Quick and Dirty Derivation </strong></summary>

Assume $QR = (I - Y T Y^\top)$, where $T$ is a forced Upper Triangle Matrix

Let us relable with $QR = (I - Y U Y^\top)$;

$ LQ = (QR)^\top = R^\top (I - YUY^\top)^\top $

Recalling $$(AB)^\top = B^\top A^\top$$

=> 
$$(Y U Y^\top)^\top = (Y^\top)^\top U^\top Y^\top = Y U^\top Y^\top$$

- Relabel $R^\top$ as $L$ (the Right triangle transpose becomes Left Triangle)
- Relabel $U^\top$ to $L$ (the Upper triangle transpose becomes Lower Triangle)

we find 
$LQ = L(I - Y L Y^\top)$

this is just for intuitons, mathematicians cover your eyes...

=>
$$LQ = L(I - Y T Y^\top)$$; where $T$ is a forced Lower Triangle Matrix
</details>

> If one does not wish to imagine the full derivation this while being extremely unrigorous gives intuition

<details markdown="1">
<summary><strong>Semi-Rigourous Derivation </strong></summary>

<details markdown="1">
<summary><strong>1. Problem Setup and Decreasing order</strong></summary>

Notice the decreasing order of the product:

$$Q := \prod_{i=1}^{n} (I - \tau_{n-i} v_{n-i} v_{n-i}^\top)$$

$$Q := (I - \tau_{n-1} v_{n-1} v_{n-1}^\top) \cdots (I - \tau_0 v_0 v_0^\top)$$

We want to find a compact representation of the form:

$$\prod_{i=1}^{n} (I - \tau_{n-i} v_{n-i} v_{n-i}^\top) = I - Y T Y^\top$$
</details>

<details markdown="1">
<summary><strong>2. Matrix Definitions</strong></summary>

Let's first define the matrix $Y$ in terms of the individual column vectors and $Y^\top$ as its column vectors transposed:

$$Y = \begin{bmatrix} v_0 & v_1 & \cdots & v_n \end{bmatrix}$$
,
$$Y^\top = \begin{bmatrix} v_0^\top \\ v_1^\top \\ \vdots \\ v_n^\top \end{bmatrix}$$

Now, let's define the initial partitions for our progressive steps:

$$Y_0 = \begin{bmatrix} v_0 \end{bmatrix}$$

and

$$Y_1 = \begin{bmatrix} v_0 & v_1 \end{bmatrix}$$
</details>

<details markdown="1">
<summary><strong>3. Base Case ($Q_0$)</strong></summary>

For $Q_0 = I - \tau_0 v_0 v_0^\top$, it should be equivalent to:

$$Q_0 = I - Y_0 T_0 Y_0^\top = I - v_0 \tau_0 v_0^\top$$

$$\implies T_0 = \begin{bmatrix} \tau_0 \end{bmatrix}$$
</details>

<details markdown="1">
<summary><strong>4. Inductive Step ($Q_1$)</strong></summary>
Then let's imagine $Q_1$:

$$Q_1 = (I - \tau_1 v_1 v_1^\top)(I - \tau_0 v_0 v_0^\top)$$

$$= I - (\tau_1 v_1 v_1^\top + \tau_0 v_0 v_0^\top - \tau_1 \tau_0 v_1 v_1^\top v_0 v_0^\top)$$

Let's think of $T_1$ as $T_0$ expanded with some unknowns recalling $T_0 = \begin{bmatrix} \tau_0 \end{bmatrix}$:

$$T_1 = \begin{bmatrix} T_0 & 0 \\ \gamma_0 & \gamma_1 \end{bmatrix}$$

$$\implies Y_1 T_1 = \begin{bmatrix} v_0 & v_1 \end{bmatrix} \begin{bmatrix} \tau_0 & 0 \\ \gamma_0 & \gamma_1 \end{bmatrix}$$

$$\implies Y_1 T_1 = \begin{bmatrix} v_0 \tau_0 + \gamma_0 v_1 & v_1 \gamma_1 \end{bmatrix}$$

$$\implies Y_1 T_1 Y_1^\top = \begin{bmatrix} v_0 \tau_0 + \gamma_0 v_1 & v_1 \gamma_1 \end{bmatrix} \begin{bmatrix} v_0^\top \\ v_1^\top \end{bmatrix} = v_0 \tau_0 v_0^\top + \gamma_0 v_1 v_0^\top + v_1 \gamma_1 v_1^\top$$

##### Solving for the Unknowns in $T_1$:

1. First, look at $\gamma_1$. This obviously needs to be $\tau_1$ because of the outer product forces:

$$\gamma_1 = \tau_1$$


2. $\gamma_0$ then must be what makes it equal, accounting for the missing term $-\tau_1 \tau_0 v_1 v_1^\top v_0 v_0^\top$:

$$\gamma_0 = -\tau_0 \tau_1 v_1^\top v_0$$


Finally, we end with our fully qualified triangular matrix:

$$T_1 = \begin{bmatrix} \tau_0 & 0 \\ -\tau_0 \tau_1 v_1^\top v_0 & \tau_1 \end{bmatrix}$$
</details>

<details markdown="1">
<summary><strong>5. General Recursion</strong></summary>
To get the full recursion, consider this block form:

$$Q_{k+1} = (I - \tau_{k+1} v_{k+1} v_{k+1}^\top)(I - Y_k T_k Y_k^\top)$$

After a bit of algebra, you'll find:

$$T_k = \begin{bmatrix} T_{k-1} & 0 \\ -\tau_k v_k^\top Y_{k-1} T_{k-1} & \tau_k \end{bmatrix}$$
</details>
</details>

> However if one wished to see a little more detail as to how the LQ / QR is actually derived

_ Tl;DR Above shows merely that the triangular matrix $T$ becomes append only which performant and provides a two derivations dependent upon the readers desire for rigour_

## LQ Summary

### Conclusion

Hopefully you can see both the intuition for how WY is derived and why I deviated from the traditional representation of QR once we consider this in the WY decomposition.

- the matrix data is in line with how its represented in newer data workflows ie Row Major and the Householder Vector
- the triangle matrix becomes append only in row major form
- helps to eliminate transposes at the boundary lines of the communication of applications and the historical Blas format
- allows us to reuse decades of optimizations simply by pivoting our representation 

All of these significantly help the memory prefether and help improve data locality.
Transposing at the boundries still hurt cost and if we can eliminate 33% of processing for small matricies we should.

It is important to note wrt to the metrics my matmul kernel approaches ~ 4/3rds the cost of Faer so it's not surprising to see the gains start to diminish.
However, I believe this is because of the kernel optimizations I am missing from my library and the immense amount of optimization that has gone into optmizing the codebase Faer.

I've merely obtained these gains by conjugating the QR decomposition with it's wanted row-major representation LQ.
After doing so all of these years of optimizations are now available in the more modern row-major form.

### Upcoming

Unfortunately, I had much more planned for this article however this article ballooned before I got to my thin Y or trapezoidal kernels which arguably more essentially in obtaining performant results.

The plan is for the following:

**Part II**
_show the SIMD trapezoidal kernels used to work around the implicit Store of Y' in the WY(LQ) and how we can use offsets and a simple FMA pattern with out kernels for direct calculation_

**Part III**
_show how one can utilize the thin q pattern within the WY and cover ideas basic ideas within Blis ie panel chunking
