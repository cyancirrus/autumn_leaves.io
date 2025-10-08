---
layout: post
title: "Implementation of Tree based Models"
date: 2025-10-08
tags: [rust, machine-learning, backend, learning]
---

**Implementations**
- [Gradient Boost](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/gradient_boost.rs)
- [Random Forest](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/random_forest.rs)
- [Decision Tree](https://github.com/cyancirrus/stellar-math/blob/main/src/learning/decision_tree.rs)

### A First Glance of the Statistical Performance

![Model Performance](./assets/tree_based_models_total_variance_explained.png)

The above chart is drawn from the **Boston Housing dataset**, which predicts the median value of houses from a handful of features. The data is small and noisy — which is *actually perfect* for testing robustness.

Each model was trained 36 times. The measure used is **Total Variance Explained (TVE)**:

Total Variance Explained (TVE), Sum Squares Error (SSE) are defined as

> $SSE(Model) := Sum (y_i - \hat{y_i})^2$

> $SSE(Data)  := Sum (y_i - mean)^2$

> $TotalVarianceExplained := 1 - \frac{SSE(model)}{SSE(Data)}$


_All TVE metrics below refer to unseen test data._

- **Random Forest — Average TVE ~78%**  
  Consistently strong across runs. The ensemble effect helps debias training data and generalize well.

- **Gradient Boosting — Average TVE ~62%**  
  Overfits heavily due to small sample size. Gradient boosting shines with large, diverse datasets. Here, it’s unstable — high variance between best and worst runs.

- **Decision Tree — Average TVE ~67%**  
  Transparent, interpretable, slightly overfits but remains solid. You can inspect node-by-node which features drive predictions.

To reproduce:
```bash
git clone https://github.com/cyancirrus/stellar-math
cargo run --example trees
open tve_chart.png
```

### Introduction

Machine learning models often emerge from a few core structures — perceptrons, decision trees, EM, clustering, and dimensionality reduction.
Here, we’ll explore how to implement three of the most classic ones:

* Decision Trees
* Random Forests
* Gradient Boosting

### Preliminaries — What *is* a Decision Tree?

Let’s start simple: a humble `if` / `else` statement.

```rust
if it_is_raining {
    items_to_carry += "umbrella";
} else {
    // no umbrella needed
}
```

That’s a one-node decision tree.
Now, imagine scaling that intuition up — predicting *wine quality*:

Features might include:

* country of origin
* year of production
* rainfall, temperature
* brand
* colour (white/red/red-blend)
* type (merlot, pinot, etc.)
* price
* **quality** ← target variable

```rust
if country == "greece" && colour == "green" {
    predicted_quality = 1000_f32;
}
```

That’s the backbone idea: a hierarchy of `if-else` splits, each narrowing down the prediction.

*Algorithmically*

```rust
let mut best_split = None;
let mut explanatory_power = 0.0;

for _ in 0..desired_number_of_nodes {
    for each_dimension in data {
        if explanatory_power(this_split) > explanatory_power {
            best_split = this_split;
            explanatory_power = explanatory_power(this_split);
        }
    }
}
```

Each node only handles its subset of data — not the entire dataset — as we recursively partition.

### Decision Tree Extensions

#### Random Forest

A **Random Forest** is just a collection of decision trees — each trained on a subsample of data or features.

```rust
let mut prediction = 0.0;
for tree in forest {
   prediction += tree.predict(unseen_data);
}
prediction /= forest.len() as f32;
```

This ensemble effect dramatically reduces variance and overfitting.

#### Gradient Boosting

Instead of averaging, **Gradient Boosting** fits each new tree to the *residual errors* of the previous one.
It’s like an iterative “error correction” process:

$prediction = y_0 + (\hat{y_0} - y_1) + (\hat{y_1} - y_2) + ... + (\hat{y_{n-1}} - y_n)$

Each new model learns to predict what its predecessors *missed*.

### Elphaba’s Look at the Great Wizard of Oz — Tree Extensions & Implementations

Everyone wants to peek behind the curtain at the “complex” methods —
and then realize they’re surprisingly elegant.

#### Gradient Boosting (Rust)

<details>
<summary>Click to expand code</summary>

{% highlight rust %}
use crate::learning::decision_tree::{DecisionTree, DecisionTreeModel};

pub struct GradientBoost {
    trees: usize,
    forest: Vec<DecisionTreeModel>,
}

impl GradientBoost {
    pub fn new(
        data: &mut Vec<Vec<f32>>,
        trees: usize,
        nodes: usize,
        obs_sample: f32,
        dim_sample: f32
    ) -> Self {
        if data.is_empty() || data[0].is_empty() {
            panic!("data is empty");
        }
        let n_obs = data[0].len();
        let dims = data.len();
        let target_idx = data.len() - 1;
        let mut sample = vec![0_f32; dims];
        let mut forest = Vec::with_capacity(trees);

        for _ in 0..trees {
            let tree = DecisionTree::new(data, obs_sample, dim_sample).train(nodes);
            for idx in 0..n_obs {
                for d in 0..dims { sample[d] = data[d][idx]; }
                let pred = tree.predict(&sample);
                data[target_idx][idx] -= pred;
            }
            forest.push(tree);
        }
        Self { trees, forest }
    }

    pub fn predict(&self, data: &[f32]) -> f32 {
        self.forest.iter().map(|t| t.predict(data)).sum()
    }
}
{% endhighlight rust %}
</details>


#### Random Forest (Rust)

<details>
<summary>Click to expand code</summary>
{% highlight rust %}
use crate::learning::decision_tree::{DecisionTree, DecisionTreeModel};

pub struct RandomForest {
    trees: usize,
    forest: Vec<DecisionTreeModel>,
}

impl RandomForest {
    pub fn new(
        data: &Vec<Vec<f32>>,
        trees: usize,
        nodes: usize,
        obs_sample: f32,
        dim_sample: f32
    ) -> Self {
        let forest = (0..trees)
            .map(|_| {
                let mut tree = DecisionTree::new(data, obs_sample, dim_sample);
                tree.train(nodes)
            })
            .collect();
        Self { trees, forest }
    }

    pub fn predict(&self, data: &[f32]) -> f32 {
        self.forest.iter().map(|t| t.predict(data)).sum::<f32>() / self.trees as f32
    }
}
{% endhighlight rust %}
</details>

See? Behind the magic — they’re just *wrappers* around the core decision tree.

### Decision Tree Implementation & Complexity

For each tree, we scan all possible splits.
Each dimension’s data is pre-sorted for efficiency.

**Initial sort cost:**
$O(d \cdot n \log n)$

**Split evaluation per node:**

$ BaseNodeSSE = SumSquares - \frac{\text{sum\_linear}^2}{\text{cardinality}} $
$ SplitNodeSSE = SSE(left) + SSE(right)$

Since we can compute these incrementally as we scan, the cost per split is linear.

**Overall cost:**

$O(d \cdot n \log n + s \cdot n \cdot d)$  
$\approx O(d \cdot n \log n)$

That’s the *coup de grâce*:
the whole algorithm’s cost is roughly that of the initial sort!

<details>
<summary>Click to expand code</summary>
{% highlight rust %}
fn delta(&self, running: &Self) -> f32 {
    if self.card == 0 || running.card == 0 || self.card  == running.card { return 0_f32 };
    let (card, l_card, r_card) = (
        self.card as f32,
        running.card as f32,
        (self.card - running.card) as f32,
    );
    let sse_curr = self.sum_squares - self.sum_linear * self.sum_linear / card;
    let sse_left = running.sum_squares - running.sum_linear * running.sum_linear / l_card;
    let sse_right = (self.sum_squares - running.sum_squares)
        - (self.sum_linear - running.sum_linear) * (self.sum_linear - running.sum_linear)
            / r_card;
    // weighted variance
    (sse_curr - sse_left - sse_right) / card
}
{% endhighlight rust %}
</details>


### Defying Gravity

Decision Trees are elegant, interpretable, and surprisingly performant.
Master the base, and the extensions — Random Forests, Gradient Boost — come almost for free.

The most powerful methods are often the simplest — once the base is solid.

Thanks for reading!

