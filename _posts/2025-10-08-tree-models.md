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

## Statistical Performance of the different Models

The following chart has been drawn from the Boston Housing Data example which for different features predicts the median value of the house. The data is a bit on the smaller types of datasets and has a bit of variance which is important when analyzing the perforamnce of the methods.

Each model was trained 36 times, as there's a bit of variance depending upon the subset that is selected (the data is incredibly small).
The measure that i've utilized is total variance explained which is defined as

```
// Sum squares error of the model is defined as

SSE(Model) := Sum (yi - yi^)^2
SSE(Data) := Sum (yi - mean)^2

// abbreviated as tve
Total Variance Explained := 1 - SSE(model) / SSE(Data);
```

[Model Performance](assets/tree_based_models_total_variance_explained.png)  

_*All TVE metrics are TVE on the unseen data from training.*_

*Random Forest* - Average TVE 78%
- Incredibly strong performance over all iterations this always has a strong perforamnce and is one of the methods which will help to debias the training data in order to help the metho perform well upon new unseen data. Average 

*Gradient Boosting* - Average TVE 62%
- Gradient boosting is normally an incredibly strong technique although the data size has incredibly defeated this method, there's not enough data variaty nor enough points in order to recursively fit the data appropriately. The model has incredibly overfit the data, parameters have not been grid searched but several have been tried. This model class is not appropriate for this dataset. Note the massive amount of variance between the lowest score for Gradient Boosted Model compared to it's Most Performant, the range is incredibly concerning and appears not stable.

*Decision Tree* - Average TVE 67%
- Base model and amazing transparency, some overfitting to the sample data which can be seen by it's relative performance to RandomForest. Can go through node by node in order to analyze which features were important. Overall a solid pick for the housing data averaging 65% Total Variance Explained for different samplings on the test dataset. Appropriate method for this problem

_In order to run the above models yourself simply clone stellar_math and run_
```bash
// see if you can find better metadata parameters!
cargo run --example trees
open tve_chart.png
```

## Decision Tree Extensions

There are two main ways that we can extend the humble decision tree

### Random Forests

A random forest is essentially a collection of decision trees. Due to the fact that we deterministically use the best split for a decision tree, we do need to subsample either/or both - the data which are consdiered, or the number of dimensions which are considered at each step.

After training we now have a large number of Decision Trees - apply named a Forest, and in order to perform a prediction
```rust 
let unseen_data = data;
let prediction = 0;
for this_tree in forest {
   prediction += this_tree.predict(unseen_data);
}
let end_prediction =  prediction / number_of_trees
end_prediction
```

Random forest helps to incredibly debias the overfitting common in decision trees and other tree based methods.


### Gradient Boosting

Gradient boosting is the other main way in order to extend decision trees. Instead of creating multiple trees and then averaging the output we recursively fit the error from the previous decision tree.

Think of it like this, we fit the first decision tree, and our prediction for the node isn't perfect, so then we look at all of our predictions and then the error. For this remaining error we then fit a new decision tree.

For the end prediction we add sum all of the decision trees output as each decision tree was fitting `y-yhat` which when we sum we unfold all of them and get
```md
prediction = y0 + ('y0 - y1) + (y1' - y2) + .. + ('yn-1 - yi);
// each y0-y1 is correcting the error of the previous iteration
prediction = y0 + error_correction[1].. + error_correction[n];
```

The predicted output isn't exactly equal, this is how the error correction term is obtained


## Elphabas Look at the Great Wizard of Oz - Tree Extensions and their Implementaitons

Everyone wishes to review the most complex methods implementations - so lets review these first.
*Gradient boosting* - lets review the complexity! - Wait ... this looks incredibly simple...
```rust
use crate::learning::decision_tree::{
    DecisionTree,
    DecisionTreeModel
}; 
pub struct GradientBoost {
    trees:usize,
    forest: Vec<DecisionTreeModel>,
}
impl GradientBoost {
    pub fn new(data:&mut Vec<Vec<f32>>, trees:usize, nodes:usize, obs_sample:f32, dim_sample:f32) -> Self {
        if data.is_empty() || data[0].is_empty() { panic!("data is empty"); }
        let n_obs = data[0].len();
        let dims = data.len();
        let target_idx = data.len()-1;
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
        Self{
            trees,
            forest,
        }
    }
    pub fn predict(&self, data:&[f32]) -> f32 {
        let mut prediction = 0_f32;
        for tree in &self.forest {
            prediction += tree.predict(data); 
        }
        prediction
    }
}
```

*Random forest* - this one must be incredibly complex! - Wait... it's exactly as trivial as the Gradient Boost implementation!
```
use crate::learning::decision_tree::{
    DecisionTree,
    DecisionTreeModel
}; 
pub struct RandomForest {
    trees: usize,
    forest: Vec<DecisionTreeModel>,
}
impl RandomForest {
    pub fn new(data:&Vec<Vec<f32>>, trees:usize, nodes:usize, obs_sample:f32, dim_sample:f32) -> Self {
        let forest:Vec<DecisionTreeModel> = (0..trees).into_iter().map(|_| {
            let mut tree = DecisionTree::new(data, obs_sample, dim_sample);
            tree.train(nodes)
        }).collect();
        Self { trees, forest }
    }
    pub fn predict(&self, data:&[f32]) -> f32 {
        let mut cumulative = 0_f32;
        for tree in &self.forest {
            cumulative += tree.predict(data);
        }
        cumulative / self.trees as f32
    }
}
```

Incredibly confusing, that's more lackluster than the reveal of Oz! They're just wrappers for the decision tree implementation. _Remember they *are* model extensions of the decision tree itself_.
Should we take a look at the Data Tree implementation?

## Decision Tree Implementation

Decision Trees actually have a fascinating implementation and their detail connects deeply to the usability of Random Forests and Gradient Boost.

### Computational Complexity

For each decision tree we must look at every potential split. Lets consider the idea - for each feature create a list of the input rows, but sorted by the dimension values themself.

Total cost for creating the sorted list for every dimension `d`.
```md
// Sort's cost is O( n log n);
O( d * n log n )
```
For this implementation I will be looking at the Sum of Squared Errors (SSE). For each dimension in order to find the best split we partition a current node's values into two nodes. As the statement is if-else, the split node will partition the data into two regions

```md
//For each member of the node

left_node := dimension_value <= partition_value;
right_node := dimension_value > partition_value;
```

We can use the following relation for variance
```
BaseNode := unsplit_node
BaseNode ::
  sum_squares // sum of y[i] * y[i] for members of this node
  sum_linear // sum of y[i] for members of the node

BaseNode SSE := sum_squares - sum_linear * sum_linear / cardinality; // cardinality is just number of members;


// This will be the new variance if we were to split at this given point
SplitNodeSSE = SSE(left_node) + SSE(right_node)

// We calculate this on a running basis
LeftNode SSE := sum_squares - sum_linear * sum_linear  / cardinality;

// And now we can derive the right nodes new SSE
inferred_sum_squares = BaseNode.sum_squares - LeftNode.sum_squares;
inferred_sum_linear = BaseNode.sum_linear - LeftNode.sum_linear;
inferred_cardinality = BaseNode.cardinality - LeftNode.cardinality;

RightNode SSE := inferred_sum_squares - inferred_sum_linear * inferred_sum_linear / inferred_cardinality; 

```

Amazing! this now means that for each potential split that our total cost for determining the optimal split is actually proporitional to the number of records!
This is _la coup de grace_ of the implementation. This is what enable the implementation to be performant and directly drives the reason for the sorted dimensions.

This means that we can simply iterate over the sorted dimensions and check if our split is optimal, linearly!
The only caveat is that when we find the new split, we simply need to sort the other dimensions. I find this incredibly beautiful and elegant, lets look at the cost.


```md
// cost for determining the optimal split for
let n := number of observations in the data;
let s := number of desired nodes // splits
let d := number of dimensions considered

// cost for all splitting of nodes
// O( find_the_best_split + sorting_other_dimensions)

O( s * n * d  + d * logn)
```

Finally we can show that the total cost for training the entire decision tree occurs
```
// O( initial_sort_dimension_cost + splitting_of_nodes )
O( d * nlog(n) + s * n * d)

// s should be << log(n)

// End Cost -- Assomptitically equal to the initial cost of sorting the dimensions!!
O( d * nlog(n));
```


## Defying Gravity

Decision Trees are amazingly powerful tools, the most important part is to ensure that a coherent and optimal decision tree base, then the implementations are more straightforward than Shiz Universities' reaction to Elphaba's green skin!


Thanks for reading!
