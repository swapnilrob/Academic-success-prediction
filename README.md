<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-ML_Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/TensorFlow-Neural_Network-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Samples-4%2C424-blue?style=for-the-badge" alt="Samples" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge" alt="License" />
</p>

<h1 align="center">🎓 Academic Success Prediction</h1>

<p align="center">
  <strong>Predict student outcomes before they happen. Intervene before it's too late.</strong>
</p>

<p align="center">
  A machine learning pipeline that classifies students as <strong>Graduate</strong>, <strong>Dropout</strong>, or <strong>Enrolled</strong> using 24 academic, demographic, and socio-economic features — comparing <strong>KNN</strong>, <strong>Logistic Regression</strong>, and a <strong>Neural Network (MLP)</strong>, with <strong>K-Means clustering</strong> for unsupervised pattern discovery. Financial factors emerged as the strongest predictors of academic outcomes.
</p>

---

## 📌 Why This Project?

Universities lose students to dropout every semester, often without understanding why until it's too late. Academic advisors work reactively — a student fails a course, misses deadlines, or simply disappears.

This project flips that model by building a **predictive classifier** that can flag at-risk students early, based on features institutions already collect:

- **Academic history** — curricular units enrolled, approved, grades across semesters
- **Financial indicators** — tuition payment status, scholarship holder, debtor flag
- **Demographic context** — age at enrollment, gender, marital status, nationality
- **Institutional data** — course, attendance mode, previous qualification

The key finding: **financial stability is the single strongest predictor of academic success** — scholarship holders graduate at significantly higher rates, while students with outstanding tuition are far more likely to drop out.

---

## 🔬 ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION                                 │
│              academic_success_dataset.xlsx → pandas DataFrame           │
│                     4,424 rows × 27 columns                             │
├─────────────────────────────────────────────────────────────────────────┤
│                     EXPLORATORY DATA ANALYSIS                           │
│     FastEDA profiling · Correlation heatmap · Class distribution        │
│       Missing value analysis · Feature importance identification        │
├─────────────────────────────────────────────────────────────────────────┤
│                         PREPROCESSING                                   │
│  Drop unnamed columns · fillna(0) · LabelEncoder (categoricals)        │
│              StandardScaler (feature normalization)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                       TRAIN / TEST SPLIT                                │
│          70% Train · 30% Test · Stratified · random_state=42            │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   KNN (k=5)  │  Logistic    │  Neural Net  │  K-Means      │
│              │  Regression  │  (MLP)       │  (k=3)        │
│  Supervised  │  Supervised  │  Supervised  │  Unsupervised  │
├──────────────┴──────────────┴──────────────┴───────────────┤
│                        EVALUATION                                       │
│   Accuracy · Precision · Recall · Confusion Matrix · ROC-AUC            │
│          Per-class ROC curves (One-vs-Rest) · Macro AUC                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| **Total samples** | 4,424 |
| **Usable after cleaning** | 3,971 |
| **Input features** | 24 (after dropping 2 unnamed empty columns) |
| **Target classes** | Graduate · Dropout · Enrolled (+ 1 from missing target values) |
| **Feature types** | Numerical + Categorical (label-encoded) |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Graduate | 1,979 | 44.7% |
| Dropout | 1,273 | 28.8% |
| Enrolled | 719 | 16.3% |
| Missing (filled as 0) | 453 | 10.2% |

> **Note:** Class imbalance is present — the Graduate class has nearly 3× the samples of Enrolled.

---

## 🧠 Models & Architecture

### K-Nearest Neighbors (KNN)

```python
KNeighborsClassifier(n_neighbors=5)
```

Standard distance-based classifier. Also tested with PCA-reduced features (2 components), which dropped accuracy to **0.379** — confirming that the full 24-dimensional feature space carries discriminative information that 2 principal components cannot capture.

### Logistic Regression

```python
LogisticRegression(max_iter=3000)
```

Linear model with extended iterations to ensure convergence on this multi-class problem. Uses one-vs-rest (OVR) strategy internally. **Best performing model** — its linear decision boundaries generalize well on structured tabular data.

### Neural Network (MLP)

```python
Sequential([
    Dense(30, activation='relu', input_shape=(24,)),
    Dense(50, activation='relu'),
    Dense(4, activation='softmax')       # 4 classes
])
# Optimizer: Adam | Loss: categorical_crossentropy
# Epochs: 30 | Batch size: 32 | Validation split: 20%
```

Two hidden layers (30 → 50 neurons) with ReLU activation, softmax output for multi-class probability distribution. Trained with 20% validation split for early stopping monitoring.

### K-Means Clustering (Unsupervised)

```python
KMeans(n_clusters=3, random_state=42)
# Visualized via PCA(n_components=2) projection
```

Applied as an unsupervised baseline to discover natural groupings. The 3 clusters **do not perfectly align with actual target classes**, revealing that the true decision boundaries are not purely geometric — they depend on feature interactions that supervised models can learn.

---

## 📈 Results

### Model Comparison

| Model | Macro AUC | Notes |
|-------|-----------|-------|
| **Logistic Regression** | **0.641** | Best overall — handles class imbalance and structured features well |
| Neural Network (MLP) | 0.609 | Competitive but overfits slightly on this dataset size |
| KNN (k=5) | 0.587 | Reasonable but sensitive to feature scaling and dimensionality |
| KNN + PCA (2D) | 0.379 accuracy | Significant degradation — information loss from dimensionality reduction |

### Evaluation Metrics Used

- **Accuracy** — overall correctness across all classes
- **Precision & Recall** — per-class breakdown (critical for imbalanced classes like Enrolled)
- **Confusion Matrix** — heatmap visualization of true vs. predicted labels
- **ROC-AUC (One-vs-Rest)** — per-class ROC curves with macro-averaged AUC for fair multi-class comparison

### Key Takeaways

- **Logistic Regression outperforms the Neural Network** on this structured, tabular dataset — a reminder that model complexity does not guarantee better results
- **Financial features dominate** — tuition fees, scholarship status, and debtor flag show the strongest correlation with the target variable in the heatmap analysis
- **Dimensionality reduction hurts KNN** — PCA(2) loses too much discriminative information, dropping accuracy by ~20 percentage points
- **K-Means clusters ≠ target classes** — unsupervised groupings capture data geometry, not the supervised outcome labels

---

## 🔧 Technical Deep Dive

### Preprocessing Pipeline

1. **Column cleanup** — Dropped `Unnamed: 25` and `Unnamed: 26` (entirely null, likely artifacts from the Excel export)
2. **Missing value handling** — `df.fillna(0)` applied globally. ~420–483 missing values per feature column
3. **Categorical encoding** — `LabelEncoder` applied to all non-numeric columns, converting string categories to integer codes
4. **Feature scaling** — `StandardScaler` normalizes all features to zero mean and unit variance (critical for KNN's distance metric and Neural Network convergence)

### EDA Highlights

- **FastEDA library** used for automated profiling and visualization
- **Correlation heatmap** revealed financial features as top predictors
- **Class distribution bar chart** confirmed imbalance requiring stratified splitting
- **Missing value analysis** identified systematic missingness across all feature columns (~10% of samples)

### Train-Test Strategy

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
```

- **70/30 split** with fixed random state for reproducibility
- **Stratified** to preserve class proportions in both train and test sets

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.7+**
- **Jupyter Notebook** or **JupyterLab**

### 1. Clone the Repository

```bash
git clone https://github.com/swapnilrob/Academic-success-prediction.git
cd Academic-success-prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn openpyxl
```

### 3. Run the Notebook

```bash
jupyter notebook 422_lab_project_G4.ipynb
```

Run all cells sequentially. The notebook handles data loading, preprocessing, model training, and evaluation in order.

### 4. Dataset

The dataset (`academic_success_dataset.xlsx`) is included in the repository. The notebook expects a CSV file — export the Excel file to CSV before running, or update the `pd.read_csv()` call to use `pd.read_excel()`:

```python
# Option A: Convert to CSV first
# Open academic_success_dataset.xlsx and Save As CSV

# Option B: Change the load line in the notebook to:
df = pd.read_excel('academic_success_dataset.xlsx')
```

---

## 📁 Project Structure

```
Academic-success-prediction/
├── 422_lab_project_G4.ipynb            # Complete ML pipeline (EDA → Models → Evaluation)
├── academic_success_dataset.xlsx       # Raw dataset (4,424 samples × 27 columns)
├── CSE422_Lab_Project_Report_G4.pdf    # Detailed project report with methodology
└── README.md                           # This file
```

---

## 👥 Team

| Name | Contributions |
|------|---------------|
| Nafisa Hasan (22299141) | Data preprocessing pipeline — missing value handling, LabelEncoder integration, StandardScaler normalization; Neural Network architecture design (layer sizing, activation functions, training hyperparameters); K-Means clustering implementation with PCA visualization |
| Swapnil Rob (22299138) | Exploratory data analysis — FastEDA profiling, correlation heatmap, class distribution analysis; KNN and Logistic Regression implementation with hyperparameter tuning; evaluation framework — confusion matrices, per-class ROC curves (OVR), macro AUC comparison |

---

## 📄 Course Context

Developed for **CSE422 — Artificial Intelligence** at **BRAC University** (Fall 2025). The project demonstrates practical application of supervised and unsupervised machine learning including data preprocessing, feature engineering, model selection and comparison, multi-class evaluation metrics (ROC-AUC, precision-recall), and deriving actionable insights from model outputs.

---

## 🧠 What We Learned

- **Simpler models win on structured data** — Logistic Regression's linear boundaries outperformed a 2-layer Neural Network on this tabular dataset, reinforcing that model selection should be driven by data characteristics, not complexity assumptions
- **Feature engineering > model complexity** — the quality of preprocessing (scaling, encoding, handling missing values) had more impact on results than switching between model architectures
- **Class imbalance requires careful evaluation** — accuracy alone is misleading when one class dominates; ROC-AUC and per-class precision/recall revealed performance gaps masked by aggregate accuracy
- **Unsupervised ≠ supervised** — K-Means clusters map data geometry, not outcome labels; the mismatch between clusters and actual classes highlights why supervised learning is essential for prediction tasks
- **Dimensionality reduction is not free** — PCA(2) destroyed KNN performance, showing that the 24-feature space carries non-redundant discriminative information
