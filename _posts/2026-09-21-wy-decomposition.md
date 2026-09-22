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
Remembering $$\hat{Y}$$ is not near-exact.

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
- Provide intuition for how blas and simd kernels can process data
- Explore the derivation for WY and show a more optimal representation of T for row major form
- Explore computational shortcuts used within my implementation of these ideas

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
<summary><strong>Glossary: Technology</strong></summary>
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

QR as a decomposition is just super handy as a way to solve a system of equations quickly, and can be used in certain places as a way to drastically simplify problems.
It's a non-iterative algorithm so it runs in dertiministic time as compared to SVD and it's constant factors for inference are near on par with SVD.

QR is one of the major tool for any industry which uses mathematics as a vehicle for insights or information, signal processing, downscaling for learning and efficiency.

## LQ and the Row Major Form

### Row Major Definition and Motivation

**Glossary of Refresher for this Article Subsetion**

<details markdown="1">
<summary><strong>Matrix Primitives</strong></summary>
> **Matrix** := A "2dimensional" grid of numbers which have coherency in it's organization ie `observations X features`  
> **Cols** := A col eg column, is a slice of data from a matrix which pertains to all features for an individual  
> **Transpose** := Take a matrix and flip it across the diagonal, ie interpretation of rows becomes cols and cols becomes rows $$\forall i,j \epsilon A_{ij} <- A_{ji}$$  
</details>

<details markdown="1">
<summary><strong>Glossary: Technology</strong></summary>
> **Blas** := Basic Linear Algebra Subprogram (B.L.A.S.) foundational project for all of numerical and computational science started in 1970s still innovations today  
> **Fortran** := A programming language in which much of Blas written in 1970s, prior to then being ported to the programming language `C` the standards of which still define current archetecture  
> **Vector** := A vector is memory on the `heap` ie not `cache` where we allocate and pass around the `pointer` ie the `reference`  
> **Vector of Vectors** := A representation of matrix which appears like `vec![vec![row1], ..., vec![rowm]]` expensive because of indirection  
> **Row Major Form** := A linearized representation of matrix which appearing as `vec![a00, a01, ..., a0n, ..., am0, ... amn]` example below  
> **Col Major Form** := A linearized representation of matrix which appearing as `vec![a00, a10, ..., am0, ..., a0n, ... amn]` example below  
> **SIMD** := Same Instruction Multiple Data (S.I.M.D.) refers to advanced vector instructions which allow for speed of numerical computation via parallelization  
> **Kernel** := A specific computational program which processes subparts of `matrix computation` via `SIMD` and advanced vector instructions which allow for parallelization  
</details>

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

Even if the transpoistion can be `SIMD'd` this is still a full scan through the data which must happen 

### TODO: Continue article and then split out into the sections - and add like article type filters
