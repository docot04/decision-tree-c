# DECISION TREE CLASSIFIER IN C

A decision tree classifier implemented completely from scratch in **C** using **ID3 algorithm**, without using any Machine Learning libraries.

**The project supports:**

- Loading categorical datasets from CSV files
- Automatic categorical value encoding
- Entropy and Information Gain calculations
- Recursive ID3 decision tree construction
- Saving trained models to a custom `.dt` file
- Prediction using a trained tree

## Project Structure

#### `data/` (Loading and managing datasets)

- Dataset structure, CSV parsing, categorical encoding, dataset allocation, cleanup

#### `train/` (Decision tree training implementation)

- TreeNode structure, entropy, information gain, tree construction, tree traversal

#### `test/` (Prediction functionality)

- encoding input values and traversing the trained tree

#### `model/` (Model persistence)

- saving and loading `.dt` model files

## USAGE

To compile:

```
cd src
gcc main.c data/dataset.c model/model.c test/prediction.c train/decision_tree.c -o dt -lm
```

To run:

1. Train the model

```
./dt train dataset.csv model.dt
```

2. To make predictions using the model

```
./dt predict model.dt "Rain,Cool,Normal,False"
```

## Working

The project implements an **ID3 (Iterative Dichotomiser 3)** decision tree.

Training workflow:

```text
                 Dataset
                    │
                    ▼
              Load CSV file
                    │
                    ▼
        Encode categorical values
                    │
                    ▼
          Calculate Entropy
                    │
                    ▼
        Calculate Information Gain
                    │
                    ▼
          Select best feature
                    │
                    ▼
          Split the dataset
                    │
                    ▼
             Recursively
            build subtrees
                    │
                    ▼
              Decision Tree
                    │
                    ▼
              Save as .dt
```

Prediction workflow:

```text
              model.dt
                  │
                  ▼
             Load model
                  │
          ┌───────┴───────┐
          ▼               ▼
      Mappings           Tree
          │               │
          └───────┬───────┘
                  ▼
           Input features
                  │
                  ▼
           Encode values
                  │
                  ▼
           Traverse tree
                  │
                  ▼
             Prediction
```

# Components

### 1. Dataset:

- The current implementation treats the final column as the **target/class column**

### 2. Categorical Encoding:

- The dataset loader automatically creates mappings for categorical values.
- The original string values are retained so that predictions can be supplied using human-readable values.

### 3. Decision Tree Algorithm:

- **Entropy** for a binary classification problem:

```text
H(S) = -p₀ log₂(p₀) - p₁ log₂(p₁)

where:
p₀ is the proportion of class 0
p₁ is the proportion of class 1
```

- **Information Gain** of a feature is:

```text
IG(S,A) = H(S) - Σ (|Sᵥ| / |S|) H(Sᵥ)

where each Sᵥ is the subset of records having a particular value of feature A.
```

### 4. Tree Construction

- The tree is constructed recursively based on highest IG
- Internally the tree is represented using the `TreeNode` structure.
  - A decision node stores: `isLeaf` `featureIndex` `classLabel` `numChildren` `featureValues` `children`
  - A leaf stores the predicted class `classLabel`

### 5. Model Persistence

- The trained model is stored in a custom text-based `.dt` format with no external serialization library required.
- The model file contains two important things: **Categorical mappings** and **Tree structure**
- The mappings are stored inside `model.dt` to allow the original CSV dataset to be completely unnecessary during prediction.

# Limitations

1. **Categorical features only**
   - Numeric/continuous features are not yet handled as continuous values.

2. **Binary classification**
   - The current entropy and majority-class implementations are designed around two classes. Multi-class classification can be added by generalizing the class-counting logic.

3. **Simple CSV parsing**
   - The CSV parser currently assumes a simple CSV format. Quoted fields containing commas are not supported.

4. **Missing values**
   - Missing values are not currently treated specially
