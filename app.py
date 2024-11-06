# Employee Attrition Prediction and Insights Dashboard
# ====================================================
# **Description**: This Streamlit application predicts employee attrition and provides actionable insights based on employee data. It enables HR teams to identify at-risk employees and make data-driven decisions to improve retention.

import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

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

# Function to load CSS from a file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css('styles_streamlit.css')

# Set the maximum elements allowed for Pandas Styler to prevent truncation in large dataframes
pd.set_option("styler.render.max_elements", 1000000)

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.write("Please follow the steps sequentially to analyze your data.")

# Navigation options with step numbers
page = st.sidebar.radio("Steps", ["1. Home", "2. Data Upload", "3. Analysis", "4. About"])

# Map page names to identifiers
page_names = {
    "1. Home": "Home",
    "2. Data Upload": "Data",
    "3. Analysis": "Analysis",
    "4. About": "About"
}

current_page = page_names[page]

# Begin main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Page Content Based on Navigation
# --------------------------------

if current_page == "Home":
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

elif current_page == "Data":
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
            st.write("**Uploaded Data Preview:**")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
    elif csv_data:
        # Attempt to read the pasted CSV data
        try:
            data = pd.read_csv(io.StringIO(csv_data))
            st.success("Data pasted successfully!")
            st.write("**Pasted Data Preview:**")
            st.dataframe(data.head())
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

        st.write("**Data after handling missing columns:**")
        st.dataframe(data.head())
        # Store the data in session state for access in other pages
        st.session_state['data'] = data

        # Guidance to proceed to Analysis page
        st.info("Data has been processed successfully! Please proceed to the **Analysis** page to view predictions and insights.")

elif current_page == "Analysis":
    # Retrieve data from session state
    data = st.session_state.get('data')

    if data is not None:
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
        # Define available features
        available_numerical_features = ['Age', 'Tenure', 'Salary', 'PerformanceRating', 'DistanceFromHome', 'TrainingHours', 'JobSatisfaction', 'EngagementScore', 'WorkLifeBalance']
        available_numerical_features = [col for col in available_numerical_features if col in data.columns]
        available_categorical_features = ['Gender', 'Department', 'JobRole', 'Overtime', 'EducationLevel']
        available_categorical_features = [col for col in available_categorical_features if col in data.columns]

        # Ensure numerical columns are converted to numeric data types
        for col in available_numerical_features:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Exclude 'EmployeeID' from features if present
        if 'EmployeeID' in data.columns:
            X = data.drop(columns=['EmployeeID'])
        else:
            X = data.copy()

        # Feature preprocessing using ColumnTransformer
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
            
        # Save the models with the current version (1.5.2)
        with open('ada_model.pkl', 'wb') as f:
            pickle.dump(ada_model, f)

        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

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
        st.subheader("Current vs Predicted Attrition")

        # Calculate current and predicted attrition rates
        current_attrition_rate = data['Attrition Prediction'].mean() * 100  # Percentage of employees predicted to leave
        predicted_attrition_rate = data['Risk Percentage'].mean()  # Average risk percentage across all employees

        # Create a DataFrame for plotting
        attrition_rates = pd.DataFrame({
            'Type': ['Current', 'Predicted'],
            'Rate': [current_attrition_rate, predicted_attrition_rate]
        })

        # Create an interactive bar chart using Plotly
        fig = px.bar(attrition_rates, x='Type', y='Rate', text='Rate',
                     title='Current vs Predicted Attrition',
                     labels={'Rate': 'Percentage (%)', 'Type': 'Attrition Type'},
                     color='Type',
                     color_discrete_sequence=px.colors.sequential.PuBu)

        # Update layout for better visuals
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          textfont_color='black')  # Set text color to black for visibility
        fig.update_layout(yaxis=dict(range=[0, max(attrition_rates['Rate']) * 1.2]),
                          showlegend=False)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Step 12: Average Risk Analysis by Job Role
        # ------------------------------------------
        st.subheader("Average Risk Percentage by Job Role")

        # Group data by JobRole and calculate mean risk percentage
        avg_risk_by_role = data.groupby('JobRole')['Risk Percentage'].mean().reset_index()

        # Create an interactive horizontal bar chart using Plotly
        fig = px.bar(avg_risk_by_role, x='Risk Percentage', y='JobRole', orientation='h',
                     title='Average Risk Percentage by Job Role',
                     labels={'Risk Percentage': 'Risk Percentage (%)', 'JobRole': 'Job Role'},
                     text='Risk Percentage',
                     color='Risk Percentage',
                     color_continuous_scale=px.colors.sequential.PuBu)

        # Update layout
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          xaxis=dict(range=[0, max(avg_risk_by_role['Risk Percentage']) * 1.2]),
                          coloraxis_showscale=False)

        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          textfont_color='black')  # Set text color to black

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Step 13: Average Risk Analysis by Department
        # --------------------------------------------
        st.subheader("Average Risk Percentage by Department")

        # Group data by Department and calculate mean risk percentage
        avg_risk_by_dept = data.groupby('Department')['Risk Percentage'].mean().reset_index()

        # Create an interactive horizontal bar chart using Plotly
        fig = px.bar(avg_risk_by_dept, x='Risk Percentage', y='Department', orientation='h',
                     title='Average Risk Percentage by Department',
                     labels={'Risk Percentage': 'Risk Percentage (%)', 'Department': 'Department'},
                     text='Risk Percentage',
                     color='Risk Percentage',
                     color_continuous_scale=px.colors.sequential.PuBu)

        # Update layout
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          xaxis=dict(range=[0, max(avg_risk_by_dept['Risk Percentage']) * 1.2]),
                          coloraxis_showscale=False)

        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          textfont_color='black')  # Set text color to black

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Step 14: Average Risk Analysis by Tenure (Seniority)
        # ----------------------------------------------------
        st.subheader("Average Risk Percentage by Seniority")

        # Group data by Tenure and calculate mean risk percentage
        avg_risk_by_tenure = data.groupby('Tenure')['Risk Percentage'].mean().reset_index()

        # Create an interactive line plot using Plotly
        fig = px.line(avg_risk_by_tenure, x='Tenure', y='Risk Percentage',
                      markers=True,
                      title='Average Attrition Risk by Seniority',
                      labels={'Tenure': 'Tenure (Years)', 'Risk Percentage': 'Risk Percentage (%)'},
                      color_discrete_sequence=['#1f77b4'])  # Use a visible blue color

        # Add data labels
        fig.update_traces(text=avg_risk_by_tenure['Risk Percentage'].round(1),
                          textposition='top center',
                          texttemplate='%{text:.1f}%',
                          textfont_color='black')  # Set text color to black

        # Update marker and line styles for better visibility
        fig.update_traces(marker=dict(size=8, color='#1f77b4'),
                          line=dict(width=2, color='#1f77b4'))

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Step 15: Individual Employee Dashboard
        # --------------------------------------
        st.subheader("Individual Employee Dashboard")

        # Allow user to select an employee by their EmployeeID
        selected_employee = st.selectbox("Select Employee by ID", data['EmployeeID'].unique(), help="Select an employee to view detailed metrics.")

        # Retrieve the data for the selected employee
        employee_data = data[data['EmployeeID'] == selected_employee].iloc[0]

        # Prepare data for visualization
        metrics = ['JobSatisfaction', 'EngagementScore', 'PerformanceRating', 'WorkLifeBalance', 'Risk Percentage']
        values = [employee_data.get(metric, np.nan) for metric in metrics]

        # Create a DataFrame for plotting
        employee_metrics = pd.DataFrame({
            'Metric': metrics,
            'Value': values
        })

        # Create a horizontal bar plot using Plotly
        fig = px.bar(employee_metrics, x='Value', y='Metric', orientation='h',
                     title=f"Dashboard for Employee {selected_employee}",
                     labels={'Value': 'Value', 'Metric': 'Metric'},
                     text='Value',
                     color='Value',
                     color_continuous_scale=px.colors.sequential.Viridis)

        # Update layout
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                          textfont_color='black')  # Set text color to black

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

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
    else:
        st.warning("Please upload and process data in the **Data Upload** page before proceeding to analysis.")

elif current_page == "About":
    # About Page Content
    st.title("About This Application")

    st.write("""
    **Employee Attrition Prediction and Insights Dashboard**

    This application was developed to help organizations proactively manage employee attrition by leveraging data analytics and machine learning. By identifying patterns and risk factors associated with employee turnover, businesses can implement targeted strategies to improve retention.

    **Key Features:**

    - Predicts individual employee attrition risk using advanced machine learning models.
    - Provides actionable insights and recommendations based on data analysis.
    - Interactive visualizations to explore trends and patterns within the organization.
    - User-friendly interface for uploading data and customizing analysis parameters.

    **Developed By:**

    - 

    **Disclaimer:**

    This tool is intended for informational purposes only and should be used as a supplement to professional HR practices. Predictions and insights are based on the data provided and the accuracy of the models used.
    """)

# End main container
st.markdown('</div>', unsafe_allow_html=True)
