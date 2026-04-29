#  Academic Success Prediction

This project was developed as part of the **CSE422: Artificial Intelligence** course at **BRAC University** in the semester of **Fall-2025**.

The goal of this project is to predict a student's academic outcome using machine learning techniques based on academic, demographic, and socio-economic features.

---

##  Problem Statement

The model predicts student outcomes into the following categories:
- Dropout
- Enrolled
- Graduate

This can help institutions identify at-risk students and provide early support such as academic assistance or financial aid.

---

##  Dataset

- Dataset size: ~4424 rows (cleaned to ~3971 usable samples)
- Features: 24 input features + 1 target variable
- Target classes:
  - Graduate
  - Dropout
  - Enrolled
  - (Additional class from missing values during preprocessing)

The dataset contains both **numerical and categorical features**, which were encoded before training.

---

##  Project Workflow

### 1. Data Preprocessing
- Removed irrelevant columns
- Handled missing values (filled with 0)
- Encoded categorical variables
- Feature scaling applied

### 2. Exploratory Data Analysis (EDA)
- Correlation heatmap analysis
- Class distribution visualization
- Missing value analysis

 Financial factors (tuition, scholarship, debtor status) showed strong influence on outcomes :contentReference[oaicite:0]{index=0}

---

### 3. Train-Test Split
- 70% Training
- 30% Testing
- Stratified split to handle class imbalance

---

##  Models Used

###  Supervised Learning
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Neural Network (MLP)

###  Unsupervised Learning
- K-Means Clustering (k=3)

---

##  Results

- **Best Model:** Logistic Regression
- Neural Network performed closely
- KNN performed comparatively lower

### Key Metrics:
- Accuracy
- Precision & Recall
- ROC-AUC

Logistic Regression achieved the highest overall performance and handled class imbalance better :contentReference[oaicite:1]{index=1}

---

## Challenges

- Class imbalance in dataset
- Handling missing values
- Choosing appropriate evaluation metrics
- Interpreting clustering results

---

##  Key Insights

- Financial stability plays a major role in academic success
- Simpler models (Logistic Regression) can outperform complex ones on structured data
- Clustering reveals patterns but doesn’t perfectly match actual outcomes

---

##  Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn

---

## Project Structure

├── 422_lab_project_G4.ipynb
├── academic_success_dataset.xlsx
├── CSE422_Lab_Project_Report_G4.pdf
└── README.md


---

## Authors

- Swapnil Rob (22299138)
- Nafisa Hasan (22299141)

---

##  Course Info

- Course: CSE422 – Artificial Intelligence
- Institution: BRAC University
