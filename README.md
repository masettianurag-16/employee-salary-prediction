# 💼 Employee Salary Prediction

An end-to-end Machine Learning regression project designed to estimate employee annual salaries based on years of experience, education level, job role, and other professional attributes.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
- [Machine Learning Model and Evaluation](#machine-learning-model-and-evaluation)
- [Project Results](#project-results)
- [How to Run the Project](#how-to-run-the-project)
- [Future Improvements](#future-improvements)

---

## 📖 Project Overview
Determining fair and competitive compensation is a critical task for human resources and organizational management. This project applies foundational Data Science and Machine Learning techniques to analyze compensation patterns and predict an employee's annual salary based on their professional background. 

This repository covers the complete data science lifecycle:
- Loading and inspecting raw tabular data
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) and data visualization
- Feature encoding and train-test splitting
- Baseline model comparison and linear regression modeling
- Model evaluation and saving trained artifacts for deployment

---

## 🎯 Problem Statement
In traditional hiring and compensation planning, setting salary bands can often be subjective, leading to internal pay discrepancies and inefficient recruitment negotiations. 

The goal of this project is to build a predictive regression model that takes an employee's professional profile (experience, role, education, etc.) and predicts their **Annual Salary in USD** (`annual_salary_usd`) with high accuracy and interpretability.

---

## 📊 Dataset Description
The model is trained on employee compensation records (`employee_salary_regression.csv`).

> **Note:** Replace placeholders below with your exact dataset statistics if known.

- **Total Records:** `[Insert total rows, e.g., 1,000 / 5,000]`
- **Total Features:** `[Insert total columns]`
- **Target Variable:** `annual_salary_usd` (Continuous numerical target)

### Feature Attributes:
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `employee_id` | Identifier | Unique ID assigned to each employee (removed during preprocessing) |
| `age` | Numerical | Age of the employee |
| `years_experience` | Numerical | Total years of relevant professional experience |
| `education_level` | Categorical | Highest education level achieved (`High School`, `Bachelor`, `Master`, `PhD`) |
| `job_role` | Categorical | Current job title (e.g., `Data Analyst`, `Software Engineer`, `ML Engineer`, `Product Manager`, `QA Engineer`, `DevOps`) |
| `city_tier` | Categorical / Ordinal | Tier rating of the employment location (`1`, `2`, `3`) |
| `performance_score` | Numerical | Employee performance rating (scale: 1.0 to 5.0) |
| `num_skills` | Numerical | Total number of relevant technical or domain skills |
| `remote_work` | Categorical / Binary | Whether the employee works remotely (`Yes` / `No`) |
| **`annual_salary_usd`** | **Numerical** | **Target variable: Annual compensation in USD** |

---

## 🛠️ Technologies Used
- **Language:** Python
- **Data Manipulation & Analysis:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Machine Learning & Preprocessing:** [Scikit-learn](https://scikit-learn.org/)
- **Model Serialization:** Joblib
- **Interactive App (Optional):** Streamlit

---

## 🔍 Data Cleaning and Exploratory Data Analysis

### 1. Data Cleaning & Preprocessing
- **Data Inspection:** Verified data types (`.info()`) and summary statistics (`.describe()`).
- **Missing Value Check:** Checked for null or missing values using `.isnull().sum()` — `[Insert finding: e.g., No missing values found / Imputed missing entries]`.
- **Duplicate Removal:** Checked for duplicate records with `.duplicated().sum()`.
- **Feature Selection:** Dropped unique identifier column `employee_id` to prevent data leakage and avoid spurious correlations.
- **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`) to transform categorical variables (`education_level`, `job_role`, `remote_work`) into numeric format for regression.

### 2. Exploratory Data Analysis (EDA) Insights
- **Salary Distribution:** Analyzed the distribution of `annual_salary_usd` using histograms to assess skewness and outliers.
- **Experience vs. Salary:** Scatter plots demonstrated a strong positive relationship between `years_experience` and `annual_salary_usd`.
- **Role & Education Analysis:** Calculated and visualized mean compensation across different job roles and education levels using bar charts.
- **Correlation Analysis:** Evaluated Pearson correlation coefficients between numerical drivers and annual salary.

---

## 🤖 Machine Learning Model and Evaluation Metrics

### Model Selection
1. **Baseline Regressor:** A `DummyRegressor(strategy="mean")` was created as a baseline benchmark to ensure the machine learning model meaningfully outperforms a simple average guess.
2. **Predictive Regressor:** A **Multiple Linear Regression** model (`LinearRegression`) was trained to estimate salary as a linear combination of numerical features and encoded categorical indicators.

### Validation Strategy
- **Train/Test Split:** 80% training set, 20% test set (`random_state=42`) to evaluate generalizability on unseen data.

### Evaluation Metrics
The model was evaluated on the unseen test set using three standard regression metrics:
- **Mean Absolute Error (MAE):** Measures the average magnitude of absolute prediction errors in USD.
- **Root Mean Squared Error (RMSE):** Measures error magnitude with higher penalization for large prediction mistakes.
- **R-squared ($R^2$) Score:** Explains the proportion of variance in annual salary captured by the model.

---

## 📈 Project Results

> **Note:** Fill in the bracketed placeholders below with your exact run outputs from the notebook.

### Model Performance vs. Baseline
| Metric | Baseline Model (`DummyRegressor`) | Linear Regression Model |
| :--- | :--- | :--- |
| **MAE** | `$[Insert Baseline MAE, e.g., 25,000]` | **`$[Insert Model MAE, e.g., 8,500]`** |
| **RMSE** | `$[Insert Baseline RMSE]` | **`$[Insert Model RMSE]`** |
| **$R^2$ Score** | `[Insert Baseline R² (typically ~0.00)]` | **`[Insert Model R² Score, e.g., 0.85]`** |

### Key Takeaways:
- The Linear Regression model significantly outperformed the baseline, demonstrating strong predictive capability.
- **Top Positive Drivers:** Features such as `years_experience`, advanced degrees (`Master`, `PhD`), and specialized roles had the largest positive coefficients on expected compensation.
- The model artifacts were serialized to disk (`salary_model.pkl` and `salary_features.pkl`) for easy integration into inference pipelines and user interfaces.

---

## 🚀 How to Run the Jupyter Notebook

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]
```

### 2. Create and Activate a Virtual Environment (Recommended)
- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
Create a `requirements.txt` file or install dependencies directly:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib streamlit
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```
Navigate to your notebook file (e.g., `employee_salary_prediction.ipynb`) and execute all cells sequentially from top to bottom.

### 5. (Optional) Run the Streamlit Web Application
If you would like to run the interactive user interface:
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements
- [ ] **Advanced Non-Linear Models:** Experiment with tree-based algorithms like Random Forest Regressor, Gradient Boosting, LightGBM, or XGBoost.
- [ ] **Hyperparameter Optimization:** Implement `GridSearchCV` or `RandomizedSearchCV` for systematic hyperparameter tuning.
- [ ] **Feature Engineering:** Create interaction terms (e.g., experience $\times$ performance score) or evaluate polynomial feature expansion.
- [ ] **Cross-Validation:** Implement K-Fold Cross-Validation to ensure robust performance across various data slices.
- [ ] **Web Deployment:** Deploy the Streamlit application to the cloud (e.g., Streamlit Community Cloud, Hugging Face Spaces, or Render).

---


