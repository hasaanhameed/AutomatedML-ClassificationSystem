# AutoML Classification System

## 🎯 Project Description
Comprehensive AutoML web application for supervised classification tasks, built with Streamlit.  
The app automates dataset upload, EDA, issue detection, preprocessing, model training (7 classifiers with hyperparameter tuning), model comparison, and report generation.

## ✨ Features
- **Automated EDA with issue detection** (missing values, outliers, correlations, distributions, class balance, constant/high-cardinality features).
- **Interactive preprocessing with user approval** for missing values, outliers, duplicates, scaling, encoding, and train–test split.
- **7 classification models with hyperparameter tuning** using GridSearchCV or RandomizedSearchCV.
- **Comprehensive model comparison dashboard** with tables and visualizations (metrics, confusion matrices, ROC curves).
- **Auto-generated detailed reports** (HTML and optional PDF via `pdfkit`).

## 🚀 Live Demo
Add your Streamlit Cloud URL here after deployment, for example:  
`https://your-streamlit-username-automl-app.streamlit.app`

## 📦 Installation

1. **Clone or copy the project**
   - Place the `automl-project` folder where you want to work.

2. **Create and activate a virtual environment (recommended)**

   ```bash
   # From inside the folder containing automl-project
   cd automl-project

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   # python -m venv .venv
   # source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > Note: PDF generation with `pdfkit` also requires the system binary `wkhtmltopdf`.  
   > If it is not installed, HTML report download will still work; PDF download may be disabled or empty.

## 💻 Usage

1. **Run the Streamlit app**

   ```bash
   cd automl-project
   streamlit run app.py
   ```

2. **Open in browser**
   - Streamlit will print a local URL such as `http://localhost:8501`.  
   - Open it in your browser if it does not open automatically.

3. **Step-by-step workflow**
   - **Upload Dataset**  
     - Upload a CSV file.  
     - Select the target (label) column.  
     - Review dataset metadata and class distribution.
   - **EDA**  
     - Click **Run EDA** to generate automated analyses in tabs (Missing Values, Outliers, Correlations, Distributions, Categorical).  
     - Optionally download screenshots/notes externally if needed.
   - **Preprocessing**  
     - Review detected issues and choose strategies for missing values, outliers, duplicates, constant/near-constant features, and class imbalance.  
     - Choose scaling method, encoding for categorical columns, and whether to create polynomial features.  
     - Set train–test split parameters and click **Apply All Preprocessing**.  
     - Preview processed data and download the processed CSV.
   - **Model Training**  
     - Choose hyperparameter search method (**GridSearchCV** or **RandomizedSearchCV**).  
     - Click **Train All Models** to train the 7 classifiers with cross-validation.  
     - View cross-validation F1-scores and training times.
   - **Results**  
     - Inspect the comparison table and sort by any metric.  
     - View confusion matrices and ROC curves for each model.  
     - Select your preferred best model and inspect its best hyperparameters.  
     - Download the model comparison table as CSV.
   - **Report**  
     - After completing the previous steps, go to **Report**.  
     - Click **Generate Report** to build an HTML (and optionally PDF) summary of the entire pipeline.  
     - Download the HTML/PDF report for your records or submission.

## 📊 Sample Results

- Use the included `sample_dataset.csv` (Iris-style dataset) to quickly test the pipeline.  
- Try:
  - Binary and multiclass datasets.
  - Datasets with missing values and outliers.
  - Larger datasets to see training time differences across models.

You can also take screenshots of:
- Model comparison table.
- Confusion matrices and ROC curves.
- EDA tabs (missing values, distributions, categorical analysis).

## 👥 Team Members
- Student 1  
- Student 2  
- Student 3  

Update the names above with your actual team members.

## 📝 License
Academic project for CS-245 Machine Learning.  
Use and modify freely for educational purposes.



