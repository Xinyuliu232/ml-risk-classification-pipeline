
# ML Risk Classification Pipeline (R)

An end-to-end **binary classification** pipeline in **R** for high-dimensional tabular data.
The workflow includes **data alignment → train/test split (stratified) → LASSO feature selection → multi-model training → metric reporting → ROC visualization**.

> **Note**: The current run shows near-perfect performance for several models. This README reports the observed results *as produced by the provided script* and also documents potential reasons (e.g., small test set) so results are interpretable and reproducible.

---

## 1) What this project does

Given:

* A feature matrix (samples × features) loaded from `Dataset_clean.txt`
* A sample metadata file with labels (`sample.txt`, binary outcome)
* (Optional) a hub gene list `hup_gene.txt` (loaded; not required for the current modeling steps)

This pipeline:

1. Aligns sample IDs between expression/features and labels
2. Splits data into train/test with **stratification** (`caret::createDataPartition`)
3. Uses **LASSO logistic regression** (`glmnet`, `alpha=1`) on the training set to select features
4. Trains multiple models on the selected feature subset
5. Evaluates models on the held-out test set using:

   * Accuracy (ACC)
   * F1 score (F1)
   * Sensitivity (Sens)
   * Specificity (Spec)
   * Area Under ROC Curve (AUC)
6. Outputs metrics and a ROC plot to `results/`

---

## 2) Methods (mapped to the script)

### 2.1 Train/test split

* Outcome `y` is a factor derived from `sample.txt`
* Train/test split uses `createDataPartition(y, p=0.8)` to preserve class proportions

### 2.2 Feature selection: LASSO (L1-regularized logistic regression)

* Model: binomial logistic regression with **L1 penalty**
* Implementation: `cv.glmnet(..., family="binomial", alpha=1)`
* Features selected at `lambda.min` are used to subset both train and test:

  * `train_X_sel`, `test_X_sel`

### 2.3 Models compared

* Logistic Regression (baseline): `glm(..., family=binomial)`
* Random Forest: `randomForest(..., ntree=300)`
* SVM (RBF kernel): `e1071::svm(..., kernel="radial", probability=TRUE)`
* XGBoost: `xgb.train(objective="binary:logistic", ...)`
* MLP (1 hidden layer): `nnet::nnet(size=16, decay=5e-4, ...)`

---

## 3) Results (current run)

### 3.1 ROC curves

![ROC Curves](results/roc_curves.png)

### 3.2 Test-set performance summary

Metrics were computed on the held-out test set using the pipeline’s `get_metrics()` function (confusion matrix + ROC/AUC).

|        Model |    ACC |     F1 |   Sens |   Spec |    AUC |
| -----------: | -----: | -----: | -----: | -----: | -----: |
|     Logistic | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
|          SVM | 0.0000 |     NA | 0.0000 | 0.0000 | 1.0000 |
|      XGBoost | 0.9167 | 0.8889 | 1.0000 | 0.8750 | 0.9375 |
|          MLP | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### 3.3 Notes on interpretation

* **Multiple models achieving ACC/AUC=1** can occur when the test set is small, the classes are highly separable

* **SVM shows ACC=0 with AUC=1**, which typically indicates a **threshold / label-probability mapping issue** rather than the ranking being wrong:

  * AUC depends on probability ranking, not the 0.5 threshold class assignment
  * ACC/F1 depend on the hard class call at a chosen threshold
    Recommended follow-ups:
  * Confirm the positive class ordering (`levels(y)`) and which probability column is being used
  * Consider using `caret::train(..., metric="ROC")` with consistent class labels and probability extraction
  * Calibrate the decision threshold (not always 0.5)

---

## 4) How to run

### Requirements

R packages used:

* dplyr, ggplot2, pROC, caret, glmnet, randomForest, e1071, xgboost, nnet

### Run

1. Place input files in the project directory:

   * `Dataset_clean.txt`
   * `sample.txt`
   * `hup_gene.txt` (optional for current pipeline)
2. Run:

   * `ML_pipeline.R`
3. Outputs:

   * `results/model_performance.txt`
   * `results/roc_curves.png`

---

## 5) Reproducibility

* Random seed is fixed (`set.seed(123)`)
* Metrics are reported on a held-out test split
* All outputs are saved under `results/`

