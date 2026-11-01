#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
df = pd.read_csv(
    r"C:\Users\ABHILASH\Downloads\employee_salary_regression.csv"
)
df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.duplicated().sum()


# In[11]:


df = df.drop(columns=["employee_id"])
df.head()


# In[12]:


import matplotlib.pyplot as plt
df["annual_salary_usd"].hist(bins=20)
plt.title("Distribution of Employee Salaries")
plt.xlabel("Annual Salary (USD)")
plt.ylabel("Number of Employees")
plt.show()


# In[13]:


plt.scatter(df["years_experience"], df["annual_salary_usd"])
plt.title("Experience vs Annual Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Annual Salary (USD)")
plt.show()


# In[14]:


df.groupby("education_level")["annual_salary_usd"].mean()


# In[15]:


import matplotlib.pyplot as plt

avg_salary = df.groupby("education_level")["annual_salary_usd"].mean()

avg_salary.plot(kind="bar")

plt.title("Average Salary by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Average Annual Salary (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[16]:


avg_role_salary = df.groupby("job_role")["annual_salary_usd"].mean()

avg_role_salary = avg_role_salary.sort_values(ascending=False)

avg_role_salary.plot(kind="bar")

plt.title("Average Salary by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Average Annual Salary (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[17]:


df[["years_experience", "annual_salary_usd"]].corr()


# In[18]:


X = df.drop(columns=["annual_salary_usd"])
y = df["annual_salary_usd"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# In[19]:


X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

X_encoded.head()


# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# In[21]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[22]:


model.fit(X_train, y_train)

print("Model training completed!")


# In[23]:


y_pred = model.predict(X_test)

print(y_pred[:10])


# In[24]:


comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

comparison.head(10)


# In[25]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# In[26]:


plt.scatter(y_test, y_pred)

plt.xlabel("Actual Salary (USD)")
plt.ylabel("Predicted Salary (USD)")
plt.title("Actual vs Predicted Salaries")

plt.show()


# In[27]:


from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)

print("Linear Regression MAE:", round(mae, 2))
print("Baseline MAE:", round(baseline_mae, 2))


# In[28]:


coefficients = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": model.coef_
})

coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()

coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
).head(10)


# In[29]:


get_ipython().run_line_magic('pip', 'install joblib')


# In[30]:


import joblib

joblib.dump(model, "salary_model.pkl")
joblib.dump(X_encoded.columns.tolist(), "salary_features.pkl")

print("Model saved successfully!")


# In[31]:


import os

print(os.getcwd())
print(os.listdir())


# In[33]:


print(X.columns.tolist())
print(X_encoded.columns.tolist())


# In[1]:


import os

print("Current folder:", os.getcwd())
print("\nFiles in this folder:")
print(os.listdir())


# In[1]:


get_ipython().run_cell_magic('writefile', 'C:\\Users\\ABHILASH\\app.py', '\nimport streamlit as st\nimport pandas as pd\nimport joblib\n\nMODEL_PATH = r"C:\\Users\\ABHILASH\\salary_model.pkl"\nFEATURE_PATH = r"C:\\Users\\ABHILASH\\salary_features.pkl"\n\nmodel = joblib.load(MODEL_PATH)\nfeature_names = joblib.load(FEATURE_PATH)\n\nst.set_page_config(\n    page_title="Employee Salary Prediction",\n    page_icon="💼"\n)\n\nst.title("Employee Salary Prediction")\nst.write("Enter employee details to estimate annual salary.")\n\nwith st.form("salary_form"):\n\n    age = st.number_input(\n        "Age", min_value=18, max_value=65, value=25\n    )\n\n    experience = st.number_input(\n        "Years of Experience",\n        min_value=0, max_value=40, value=2\n    )\n\n    education = st.selectbox(\n        "Education Level",\n        ["Bachelor", "High School", "Master", "PhD"]\n    )\n\n    job_role = st.selectbox(\n        "Job Role",\n        [\n            "Data Analyst",\n            "DevOps",\n            "ML Engineer",\n            "Product Manager",\n            "QA Engineer",\n            "Software Engineer"\n        ]\n    )\n\n    city_tier = st.selectbox("City Tier", [1, 2, 3])\n\n    performance = st.number_input(\n        "Performance Score",\n        min_value=1.0, max_value=5.0, value=3.0\n    )\n\n    num_skills = st.number_input(\n        "Number of Skills",\n        min_value=1, max_value=20, value=5\n    )\n\n    remote_work = st.selectbox(\n        "Remote Work", ["No", "Yes"]\n    )\n\n    submitted = st.form_submit_button("Predict Salary")\n\nif submitted:\n\n    employee = pd.DataFrame([{\n        "age": age,\n        "years_experience": experience,\n        "education_level": education,\n        "job_role": job_role,\n        "city_tier": city_tier,\n        "performance_score": performance,\n        "num_skills": num_skills,\n        "remote_work": 1 if remote_work == "Yes" else 0\n    }])\n\n    encoded = pd.get_dummies(\n        employee, drop_first=True, dtype=int\n    )\n\n    encoded = encoded.reindex(\n        columns=feature_names, fill_value=0\n    )\n\n    prediction = model.predict(encoded)[0]\n\n    st.success(\n        f"Estimated Annual Salary: ${prediction:,.2f}"\n    )\n\n    st.caption(\n        "This is an estimate, not a guaranteed salary."\n    )\n')


# In[ ]:


get_ipython().system('python -m streamlit run "C:\\Users\\ABHILASH\\app.py"')

