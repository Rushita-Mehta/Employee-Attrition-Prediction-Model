# Employee Attrition Prediction and Insights Dashboard
# ====================================================
# **Description**: This Streamlit application predicts employee attrition and provides actionable insights based on employee data. It enables HR teams to identify at-risk employees and make data-driven decisions to improve retention.

import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set the maximum elements allowed for Pandas Styler to prevent truncation in large dataframes
pd.set_option("styler.render.max_elements", 1000000)

# Function Definitions
# --------------------

def highlight_scores(val, max_score):
    """
    Highlights cell background color based on score values to emphasize low and high scores.

    Parameters:
    - val (float): The value to evaluate.
    - max_score (int): The maximum possible score for the metric.

    Returns:
    - str: A string representing the CSS style to apply to the cell.
    """
    try:
        val = float(val)
        color = ''
        # For scores out of 10 (e.g., Job Satisfaction)
        if max_score == 10:
            if val < 2:
                color = 'red'    # Very low score
            elif val > 8:
                color = 'green'  # Very high score
        # For scores out of 5 (e.g., Performance Rating, Work-Life Balance)
        elif max_score == 5:
            if val < 2:
                color = 'red'
            elif val > 4:
                color = 'green'
        return f'background-color: {color}'
    except:
        return ''

def color_risk(val):
    """
    Applies background color to the 'Risk Category' column based on the risk level.

    Parameters:
    - val (str): The risk category ('Low Risk', 'Medium Risk', 'High Risk').

    Returns:
    - str: A string representing the CSS style to apply to the cell.
    """
    color = ''
    if val == 'Low Risk':
        color = 'green'
    elif val == 'Medium Risk':
        color = 'orange'
    else:
        color = 'red'
    return f'background-color: {color}'

# Step 1: Application Title and Overview
# --------------------------------------
st.title("Employee Attrition Prediction and Insights Dashboard")

# Provide a detailed description of the application and its purpose
st.write("""
**Overview**

This application allows businesses to predict employee attrition and gain actionable insights from their employee data. By analyzing key attributes, the model identifies employees who are at risk of leaving the company. This enables HR teams and managers to take proactive measures to improve employee retention, reduce turnover costs, and enhance overall organizational performance.

**Attrition Prediction Scores**:
- **0**: The employee is predicted to stay with the company.
- **1**: The employee is predicted to leave (high risk of attrition).
""")

# Step 2: Data Upload Section
# ---------------------------
st.subheader("Upload Employee Data or Paste CSV Data")

st.write("""
**Data Requirements**

For accurate predictions, the dataset should include the following columns:

- **EmployeeID**: Unique identifier for each employee.
- **Age**: Age of the employee.
- **Gender**: Gender of the employee.
- **Department**: Department where the employee works.
- **Tenure**: Number of years the employee has been with the company.
- **JobRole**: Role or position of the employee within the company.
- **Salary**: Current salary of the employee.
- **JobSatisfaction**: Self-reported job satisfaction score (out of 10).
- **EngagementScore**: Employee engagement score (percentage).
- **PerformanceRating**: Performance rating (out of 5).
- **WorkLifeBalance**: Work-life balance rating (out of 5).
""")

# Provide options for uploading data via file or pasting CSV data
uploaded_file = st.file_uploader("Upload CSV File", type="csv", help="Upload your employee data file in CSV format.")
csv_data = st.text_area("Or paste your CSV data here (CSV format)", height=200, help="Paste CSV data if you don't want to upload a file.")

# Step 3: Data Loading and Preprocessing
# --------------------------------------
data = None  # Initialize the data variable
missing_columns = []  # List to keep track of missing columns

# Check if a file was uploaded or CSV data was pasted
if uploaded_file:
    # Attempt to read the uploaded CSV file
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("**Uploaded Data Preview:**", data.head())
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
elif csv_data:
    # Attempt to read the pasted CSV data
    try:
        data = pd.read_csv(io.StringIO(csv_data))
        st.success("Data pasted successfully!")
        st.write("**Pasted Data Preview:**", data.head())
    except Exception as e:
        st.error(f"Error reading pasted CSV data: {e}")

# Step 4: Handling Missing Columns and Data Imputation
# ----------------------------------------------------
if data is not None:
    # Define key columns required for the model
    key_columns = ['JobSatisfaction', 'EngagementScore', 'WorkLifeBalance']
    # Identify any missing key columns
    missing_columns = [col for col in key_columns if col not in data.columns]

    if missing_columns:
        st.warning(f"The following key columns are missing: {', '.join(missing_columns)}. "
                   "This may reduce the prediction accuracy. "
                   "We are imputing default values to allow the model to make predictions.")

        # Impute missing columns with default neutral values
        for col in missing_columns:
            if col == 'JobSatisfaction':
                data[col] = 5  # Neutral score out of 10
            elif col == 'EngagementScore':
                data[col] = 50  # Neutral engagement score (percentage)
            elif col == 'WorkLifeBalance':
                data[col] = 3  # Neutral score out of 5

    # Define the list of available numerical and categorical features based on the data
    available_numerical_features = ['Age', 'Tenure', 'Salary', 'PerformanceRating', 'DistanceFromHome', 'TrainingHours'] + key_columns
    available_numerical_features = [col for col in available_numerical_features if col in data.columns]
    available_categorical_features = ['Gender', 'Department', 'JobRole', 'Overtime', 'EducationLevel']
    available_categorical_features = [col for col in available_categorical_features if col in data.columns]

    # Ensure numerical columns are converted to numeric data types
    for col in available_numerical_features:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Step 5: Model Selection and Prediction Threshold
    # ------------------------------------------------
    st.subheader("Model Selection and Prediction")

    # Provide detailed explanations for each model option
    st.write("""
    **Model Selection**

    Choose a predictive model based on your business needs:

    - **AdaBoost**: Ideal for businesses needing quick, actionable results with simpler data. AdaBoost is efficient and provides good predictions with a moderate number of features. It is suitable when you need to rapidly identify at-risk employees for early intervention.

    - **Random Forest**: Suitable for complex data patterns and provides deeper insights. Random Forest excels with larger datasets and helps understand trends and factors contributing to attrition. Choose this model if you seek comprehensive insights into employee behavior.
    """)

    # Model selection dropdown
    model_choice = st.selectbox("Choose a model for predicting employee attrition:", ["AdaBoost", "Random Forest"])

    # Explain the concept of prediction threshold
    st.write("""
    **Prediction Threshold**

    The prediction threshold determines the sensitivity of the model in classifying employees as at risk of attrition.

    - A **lower threshold** captures more employees who might be at risk, allowing for early intervention but may include false positives.
    - A **higher threshold** focuses on employees who are highly likely to leave, reducing false positives but potentially missing early warning signs.

    Adjust the threshold based on your organization's risk tolerance and resource capacity for intervention.
    """)

    # Slider to select prediction threshold
    threshold = st.slider("Select Prediction Threshold", 0.0, 1.0, 0.4, step=0.01)

    # Step 6: Data Preparation for Prediction
    # ---------------------------------------
    # Exclude 'EmployeeID' from features if present
    if 'EmployeeID' in data.columns:
        X = data.drop(columns=['EmployeeID'])
    else:
        X = data.copy()

    # Feature preprocessing using ColumnTransformer
    # - Standardize numerical features
    # - One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), available_categorical_features)
        ],
        remainder='drop'  # Drop any features not specified in the transformers
    )

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Step 7: Load Pre-trained Models
    # -------------------------------
    # Load the pre-trained AdaBoost and Random Forest models from pickle files
    with open('ada_model.pkl', 'rb') as f:
        ada_model = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    # Select the model based on user choice
    model = ada_model if model_choice == "AdaBoost" else rf_model

    # Step 8: Making Predictions
    # --------------------------
    # Predict the probabilities of attrition for each employee
    probabilities = model.predict_proba(X_processed)[:, 1]  # Probability of the positive class (attrition)

    # Convert probabilities to percentages
    risk_percentage = probabilities * 100

    # Map risk percentages to risk categories
    risk_category = np.where(risk_percentage < 30, 'Low Risk',
                             np.where(risk_percentage < 70, 'Medium Risk', 'High Risk'))

    # Generate binary predictions based on the selected threshold
    predictions = (probabilities >= threshold).astype(int)

    # Step 9: Adding Predictions to the DataFrame
    # -------------------------------------------
    # Add the predictions and risk assessments to the original data
    data['Attrition Prediction'] = predictions  # 0 or 1 indicating stay or leave
    data['Risk Percentage'] = risk_percentage.round(1)  # Rounded to one decimal place
    data['Risk Category'] = risk_category  # Categorical risk assessment

    # Step 10: Displaying Prediction Results
    # --------------------------------------
    st.subheader("Attrition Prediction Results")

    # Style the DataFrame for better visualization
    styled_df = data.style.applymap(
        lambda val: highlight_scores(val, 10),
        subset=['JobSatisfaction'] if 'JobSatisfaction' in data.columns else []
    ).applymap(
        lambda val: highlight_scores(val, 5),
        subset=['PerformanceRating', 'WorkLifeBalance'] if 'WorkLifeBalance' in data.columns else []
    ).applymap(
        color_risk, subset=['Risk Category']
    )

    # Display the styled DataFrame
    st.dataframe(styled_df)

    # Step 11: Visualizing Current vs Predicted Attrition
    # ---------------------------------------------------
    st.subheader("Current Attrition vs Predicted Attrition")

    # Calculate current and predicted attrition rates
    current_attrition_rate = data['Attrition Prediction'].mean() * 100  # Percentage of employees predicted to leave
    predicted_attrition_rate = data['Risk Percentage'].mean()  # Average risk percentage across all employees

    # Create a simple line plot to compare current and predicted attrition rates
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(['Current', 'Predicted'], [current_attrition_rate, predicted_attrition_rate],
            marker='o', color='blue')
    ax.set_title('Current vs Predicted Attrition', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=10)

    # Annotate the plot with the actual values
    for i, value in enumerate([current_attrition_rate, predicted_attrition_rate]):
        ax.text(i, value + 0.5, f'{value:.1f}%', ha='center', fontsize=9, color='grey')

    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('darkgrey')
    ax.spines['bottom'].set_color('darkgrey')

    # Display the plot
    st.pyplot(fig)

    # Step 12: Average Risk Analysis by Job Role
    # ------------------------------------------
    st.subheader("Average Risk Percentage by Job Role")

    # Group data by JobRole and calculate mean risk percentage
    avg_risk_by_role = data.groupby('JobRole')['Risk Percentage'].mean().reset_index()

    # Create a horizontal bar plot for average risk by job role
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Risk Percentage', y='JobRole', data=avg_risk_by_role,
                palette="Blues_d", ax=ax)
    ax.set_title('Average Risk Percentage by Job Role', fontsize=11)

    # Annotate each bar with the risk percentage
    for i in range(len(avg_risk_by_role)):
        ax.text(avg_risk_by_role['Risk Percentage'][i] + 0.5, i,
                f'{avg_risk_by_role["Risk Percentage"][i]:.1f}%', va='center',
                fontsize=9, color='grey')

    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('darkgrey')
    ax.spines['bottom'].set_color('darkgrey')

    # Display the plot
    st.pyplot(fig)

    # Step 13: Average Risk Analysis by Department
    # --------------------------------------------
    st.subheader("Average Risk Percentage by Department")

    # Group data by Department and calculate mean risk percentage
    avg_risk_by_dept = data.groupby('Department')['Risk Percentage'].mean().reset_index()

    # Create a horizontal bar plot for average risk by department
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Risk Percentage', y='Department', data=avg_risk_by_dept,
                palette="Blues_d", ax=ax)
    ax.set_title('Average Risk Percentage by Department', fontsize=11)

    # Annotate each bar with the risk percentage
    for i in range(len(avg_risk_by_dept)):
        ax.text(avg_risk_by_dept['Risk Percentage'][i] + 0.5, i,
                f'{avg_risk_by_dept["Risk Percentage"][i]:.1f}%', va='center',
                fontsize=9, color='grey')

    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('darkgrey')
    ax.spines['bottom'].set_color('darkgrey')

    # Display the plot
    st.pyplot(fig)

    # Step 14: Average Risk Analysis by Tenure (Seniority)
    # ----------------------------------------------------
    st.subheader("Average Risk Percentage by Seniority")

    # Group data by Tenure and calculate mean risk percentage
    avg_risk_by_tenure = data.groupby('Tenure')['Risk Percentage'].mean().reset_index()

    # Create a line plot to show risk percentage over tenure
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x='Tenure', y='Risk Percentage', data=avg_risk_by_tenure,
                 marker="o", ax=ax, color="blue")

    # Annotate each point with the risk percentage
    for i in range(len(avg_risk_by_tenure)):
        ax.text(avg_risk_by_tenure['Tenure'][i], avg_risk_by_tenure['Risk Percentage'][i] + 0.5,
                f'{avg_risk_by_tenure["Risk Percentage"][i]:.1f}%', ha='center',
                fontsize=9, color='grey')

    # Set labels and title
    ax.set_xlabel('Tenure (Years)')
    ax.set_title('Average Attrition Risk by Seniority', fontsize=11)

    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('darkgrey')
    ax.spines['bottom'].set_color('darkgrey')

    # Display the plot
    st.pyplot(fig)

    # Step 15: Individual Employee Dashboard
    # --------------------------------------
    st.subheader("Individual Employee Dashboard")

    # Allow user to select an employee by their EmployeeID
    selected_employee = st.selectbox("Select Employee by ID", data['EmployeeID'].unique())

    # Retrieve the data for the selected employee
    employee_data = data[data['EmployeeID'] == selected_employee].iloc[0]

    # Prepare data for visualization
    metrics = ['JobSatisfaction', 'EngagementScore', 'PerformanceRating', 'WorkLifeBalance', 'Risk Percentage']
    values = [employee_data.get(metric, np.nan) for metric in metrics]

    # Create a horizontal bar plot for the employee's metrics
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(metrics, values, color=['blue', 'green', 'orange', 'purple', 'red'])

    # Annotate each bar with the metric value
    for i, value in enumerate(values):
        ax.text(value + 0.5, i, f'{value:.1f}', va='center', fontsize=9, color='darkgrey')

    # Set title and labels
    ax.set_title(f"Dashboard for Employee {selected_employee}", fontsize=11)

    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('darkgrey')
    ax.spines['bottom'].set_color('darkgrey')

    # Display the plot
    st.pyplot(fig)

    # Step 16: Insights and Recommendations
    # -------------------------------------
    st.subheader("Insights and Recommendations")

    # Calculate overall average risk and attrition rate
    overall_avg_risk = data['Risk Percentage'].mean()
    avg_attrition_rate = data['Attrition Prediction'].mean() * 100

    # Identify departments with minimum and maximum risk
    min_risk_dept = data.groupby('Department')['Risk Percentage'].mean().idxmin()
    max_risk_dept = data.groupby('Department')['Risk Percentage'].mean().idxmax()

    # Display key findings
    st.write(f"Based on the data, the overall average attrition risk is **{overall_avg_risk:.1f}%**.")
    st.write(f"The current predicted attrition rate is **{avg_attrition_rate:.1f}%**.")
    st.write(f"The department with the lowest attrition risk is **{min_risk_dept}**, while the highest risk is in **{max_risk_dept}**.")

    # Provide dynamic recommendations based on overall attrition risk
    st.write("**Recommendations:**")
    if overall_avg_risk < 3:
        st.write("- **Very Low Risk (0-3%)**: Attrition risk is extremely low. Maintain current employee engagement and retention strategies.")
    elif 3 <= overall_avg_risk < 9:
        st.write("- **Low Risk (3-9%)**: Risk is low but monitor for early signs of disengagement. Consider periodic check-ins and employee feedback sessions.")
    elif 9 <= overall_avg_risk < 12:
        st.write("- **Moderate Risk (9-12%)**: There is a noticeable increase in risk. Conduct employee surveys to understand potential issues affecting satisfaction and engagement.")
    elif 12 <= overall_avg_risk < 20:
        st.write("- **Moderate to High Risk (12-20%)**: Implement recognition programs and provide career growth opportunities to improve morale and reduce attrition risk.")
    elif 20 <= overall_avg_risk < 30:
        st.write("- **High Risk (20-30%)**: Immediate attention is required. Evaluate compensation packages, benefits, and workloads to address employee concerns.")
    else:
        st.write("- **Critical Risk (>30%)**: The organization is at serious risk of high attrition. Initiate company-wide efforts focused on improving employee satisfaction and engagement.")

    # Step 17: Additional Data-Driven Insights
    # ----------------------------------------
    st.write("""
    **Data-Driven Insights**

    The following insights have been derived from the data to help inform strategic actions:
    """)

    # Insight 1: Correlation between Engagement Score and Attrition Risk
    if 'EngagementScore' in data.columns:
        correlation = data['EngagementScore'].corr(data['Risk Percentage'])
        st.write(f"1. **Engagement and Attrition Risk Correlation**: There is a correlation of **{correlation:.2f}** between engagement scores and attrition risk. Lower engagement scores are associated with higher attrition risk. Enhancing employee engagement programs may help reduce attrition.")

    # Insight 2: Impact of Overtime on Attrition Risk
    if 'Overtime' in data.columns:
        overtime_risk = data.groupby('Overtime')['Risk Percentage'].mean().reset_index()
        st.write("2. **Overtime Impact**: Employees who frequently work overtime have higher attrition risk. Consider balancing workloads and encouraging a healthy work-life balance.")
        for index, row in overtime_risk.iterrows():
            st.write(f"- **Overtime = {row['Overtime']}**: {row['Risk Percentage']:.1f}% risk")

    # Insight 3: Training Hours and Attrition Risk
    if 'TrainingHours' in data.columns:
        avg_training_hours = data['TrainingHours'].mean()
        st.write(f"3. **Training and Development**: The average training hours per employee is **{avg_training_hours:.1f}**. Providing additional training and development opportunities can enhance employee satisfaction and reduce attrition risk.")

    # Step 18: Downloading the Results
    # --------------------------------
    st.subheader("Download Results")

    # Convert the data to CSV format for download
    csv = data.to_csv(index=False).encode('utf-8')

    # Provide a download button
    st.download_button(label="Download Results as CSV",
                       data=csv,
                       file_name='attrition_predictions.csv',
                       mime='text/csv')

# End of Application
