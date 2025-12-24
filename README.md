# AutoML Classification System

This project is a Streamlit-based AutoML application for supervised classification. It guides users through dataset upload, automated exploratory data analysis (EDA), preprocessing, model training with hyperparameter optimization, evaluation, and report generation through an interactive web interface.

---

## Features

### 1. Dataset Upload

- Upload CSV datasets via drag-and-drop or file browser
- Automatic dataset overview:
  - Row and column counts
  - Count of missing cells
  - Column data types
  - Summary statistics for all columns
  - Preview of first and last rows
- Target column selection (defaults to the last column)

---

### 2. Automated Exploratory Data Analysis (EDA)

- One-click EDA with progress indicator
- Tabbed EDA interface:
  - **Missing Values**
    - Detailed missing-value summary table per feature
  - **Outliers**
    - IQR-based outlier table
    - Z-score-based outlier table
  - **Correlations**
    - Correlation matrix heatmap
    - Table of highly correlated feature pairs
  - **Distributions**
    - Summary statistics and distribution plots for numeric features
  - **Categorical Features**
    - Cardinality table
    - Bar charts for each categorical feature

---

### 3. Data Issues and Preprocessing

- Automatic issue detection driven by EDA and direct checks:
  - Missing values, outliers, and other data quality signals
- Per-issue strategy selection via radio buttons, stored in session state
- Issue severity levels:
  - Critical
  - Warning
  - Informational
- Global preprocessing configuration:
  - **Feature scaling**
    - StandardScaler
    - MinMaxScaler
    - RobustScaler
    - None
  - **Categorical encoding (per column)**
    - One-Hot Encoding
    - Label Encoding
    - Ordinal Encoding
  - **Optional feature engineering**
    - Polynomial features for numeric columns
  - **Train–test split**
    - Adjustable test set size
    - Configurable random state
- Confirmation view:
  - JSON summary of selected issue-handling strategies
- Preprocessing execution:
  - Single button to apply all preprocessing with spinner
  - Processed dataset and train–test split stored in session
  - Preprocessing summary shown via JSON viewer
  - Preview of processed data
  - Download button for processed dataset (CSV)

---

### 4. Model Training and Hyperparameter Optimization

- Uses the preprocessed train–test split
- Trains seven classification models:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Naive Bayes
  - Random Forest
  - Support Vector Machine
  - Rule-Based Classifier (shallow decision tree)
- Clear listing of all models to be trained
- Hyperparameter search methods:
  - GridSearchCV
  - RandomizedSearchCV
- Informational guidance:
  - Explains that GridSearchCV may significantly increase Random Forest training time
  - Recommends RandomizedSearchCV as a faster alternative
- Training execution:
  - Single button to train all models
  - Spinner, progress bar, and status text during training
  - Stores best estimator, best parameters, training time, and cross-validated F1-score per model
- Cross-validation summary:
  - Table of model names, CV F1-scores, and training times

---

### 5. Results and Evaluation

- Automatic evaluation of all trained models on the test set with spinner
- Metrics per model:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-score (weighted)
  - ROC-AUC (binary classification, if supported)
  - Training time
  - Best hyperparameters
- Model comparison table:
  - Styled to highlight best metrics and fastest training times
- Sorting options:
  - Sort by any metric or training time
- Visual evaluation:
  - Confusion matrix heatmaps
  - ROC curves for binary classification
- Best model selection:
  - Dropdown to select preferred model
  - Stores selected model in session state
  - Displays best hyperparameters in JSON format
- Export:
  - Downloadable CSV of the model comparison table

---

### 6. Report Generation

- Requires completion of:
  - Upload
  - EDA
  - Preprocessing
  - Training
  - Results
- Collects:
  - Dataset information (rows, columns, features, target, class distribution)
  - EDA results
  - Preprocessing steps
  - Evaluation results
  - Selected best model
- One-click report generation with spinner
- Output:
  - HTML report available for download

---

### 7. UI, UX, and Theming

- Single-page Streamlit application with sidebar navigation:
  - Upload Dataset
  - EDA
  - Preprocessing
  - Model Training
  - Results
  - Report
- Dark theme with black and purple tones
  - High-contrast text for readability
  - Consistent styling across all pages
- Purple-themed plotting:
  - All Seaborn and Matplotlib plots use a consistent purple palette
- Clear loading indicators:
  - Spinners for dataset loading, EDA, preprocessing, model training, evaluation, and report generation
- Responsive layout using wide mode and column-based layouts

---



