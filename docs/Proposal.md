# Lifestyle-Based Blood Pressure Prediction

## Prepared for  
UMBC Data Science Master’s Degree Capstone  
Term: Fall 2025  
Instructor: **Dr. Chaojie (Jay) Wang**

## Author  
**Mera Mathew**

- **GitHub Repository:** https://github.com/MeraMathew/UMBC-DATA606-Capstone  
- **LinkedIn Profile:** https://www.linkedin.com/in/mera-mathew-026771170/  
- **PowerPoint Presentation:** *To be added*  
- **YouTube Video Presentation:** https://youtu.be/iThk3fUDH90  

## 2. Background

Hypertension (high blood pressure) is a leading risk factor for cardiovascular disease and stroke.  
This project develops a machine-learning regression model to **predict systolic and diastolic blood pressure** using demographic, dietary, physical activity, smoking, and alcohol indicators derived from the U.S. **National Health and Nutrition Examination Survey (NHANES)**.

**Why it matters**  
- Enables early detection of elevated blood pressure risk.  
- Highlights modifiable lifestyle factors—such as sodium intake, exercise, smoking, and drinking—that can guide personal and public-health interventions.  
- Demonstrates an end-to-end data science workflow with real public health data.

**Research Questions**  
1. To what extent can demographic, diet, physical activity, smoking, and alcohol variables predict blood pressure?  
2. Which factors contribute most to systolic and diastolic blood pressure?


## 3. Data

#### Data Sources
NHANES survey cycles covering **August 2021 – August 2023**  
(publicly available at [https://www.cdc.gov/nchs/nhanes/](https://www.cdc.gov/nchs/nhanes/)):
#### Data Shape
- Rows: 7518 
- Columns: 17
- Approx file size (MB): 1.02

To build a single machine-learning-ready table, **combined several NHANES component datasets** by joining on the unique participant ID `SEQN`:

- **Examination**: BPX (Blood Pressure), BMX (Body Measures)
- **Demographics**: DEMO
- **Dietary**: DR1TOT (Day-1 nutrient intake)
- **Physical Activity**: PAQ
- **Smoking**: SMQ
- **Alcohol**: ALQ

This integration step ensured that each participant’s demographics, body measures, diet, activity level, smoking history, and alcohol consumption are available in **one dataset**.

#### Data Details

- **Time period covered**  
  August 2021 – August 2023 (NHANES survey cycles for those two years)

- **Observation unit**  
  Each row represents **one individual NHANES participant**, with all lifestyle, demographic, and health measurements merged into a single record.

- **Data dictionary (key columns)**  

| Column | Type | Definition / Units | Categories / Encoded Labels |
|--------|------|--------------------|------------------------------|
| `Participant_ID` | int | Unique NHANES participant ID | e.g., 130378 |
| `Systolic_BP` | float | Mean systolic blood pressure (mmHg) | – |
| `Diastolic_BP` | float | Mean diastolic blood pressure (mmHg) | – |
| `Age_Years` | int | Age of participant | e.g., 43 |
| `Gender` | category | Biological sex | 0 = Male, 1 = Female |
| `Race_Ethnicity` | category | Race/ethnicity group | 0 = Non-Hispanic White, 1 = Non-Hispanic Black, 2 = Mexican American, 3 = Other (Other Hispanic, Non-Hispanic Asian, Multiracial) |
| `BMI` | float | Body Mass Index (kg/m²) | e.g., 27.5 |
| `Weight_kg` | float | Body weight | e.g., 74.0 |
| `Height_cm` | float | Body height | e.g., 172.0 |
| `Sodium_mg` | float | Daily sodium intake | e.g., 3200 |
| `Potassium_mg` | float | Daily potassium intake | e.g., 2900 |
| `Calories_kcal` | float | Total daily calorie intake | e.g., 2100 |
| `Vigorous_Activity_Days` | int | Days/week of vigorous activity | 0–7 |
| `Moderate_Activity_Days` | int | Days/week of moderate activity | 0–7 |
| `Ever_Smoked_100_Cigs` | category | Ever smoked ≥100 cigarettes | 0 = No, 1 = Yes |
| `Current_Smoking_Status` | category | Current smoking frequency | 0 = Not at all, 1 = Some days, 2 = Every day |
| `Drinks_per_Week` | float | Estimated alcoholic drinks per week | e.g., 2.5 |
| `Had_12_Drinks_Lifetime` | category | Ever consumed ≥12 drinks in lifetime | 0 = No, 1 = Yes |

- **Target / Label variables for ML model**  
  - `Systolic_BP`  
  - `Diastolic_BP`

- **Feature / Predictor candidates**  
  - Demographics: `Age_Years`, `Gender`, `Race_Ethnicity`
  - Body measures: `BMI`, `Weight_kg`, `Height_cm`
  - Diet: `Sodium_mg`, `Potassium_mg`, `Calories_kcal`
  - Physical activity: `Vigorous_Activity_Days`, `Moderate_Activity_Days`
  - Lifestyle habits: `Ever_Smoked_100_Cigs`, `Current_Smoking_Status`, `Drinks_per_Week`


## 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was performed using a Jupyter Notebook to understand the target variables—systolic and diastolic blood pressure—and their relationships with selected demographic and lifestyle features. The analysis focused on identifying data quality issues, understanding variable distributions, and extracting insights to guide feature selection for predictive modeling.

---

### 4.1 Target Variable and Feature Selection

The primary target variables for this study are:
- **Systolic Blood Pressure (Systolic_BP)**
- **Diastolic Blood Pressure (Diastolic_BP)**

The following features were retained based on clinical relevance and data availability:
- Age
- Gender
- Race/Ethnicity
- Height and Weight
- Smoking Status
- Alcohol Consumption
- Dietary intake (Sodium, Potassium, Calories)

All unrelated or redundant columns were removed to maintain a focused and interpretable dataset.

---

### 4.2 Summary Statistics

Basic descriptive statistics were computed for all key numeric variables, including blood pressure, age, height, weight, sodium, potassium, and calorie intake.

**Key observations:**
- Systolic BP ranged from **70 to 232 mmHg**, with a mean of approximately **119 mmHg**.
- Diastolic BP ranged from **34 to 139 mmHg**, with a mean of approximately **72 mmHg**.
- Dietary variables (sodium, potassium, calories) showed high variance, reflecting diverse lifestyle patterns.
- Height and weight distributions aligned with expected NHANES population characteristics.

---

### 4.3 Missing Data and Data Quality Assessment

#### Missing Values Analysis

| Column | % Missing |
|------|-----------|
| Cigarettes_Per_Day | 97% |
| Current_Smoking_Status | 67% |
| Drinks_Per_Day | 46% |
| Sodium_mg / Potassium_mg / Calories_kcal | ~22% |
| BMI / Weight / Height | <1% |

**Data cleansing actions taken:**
- Dropped `Cigarettes_Per_Day` due to excessive missingness.
- Median imputed sodium, potassium, calorie intake, height, and weight.
- Mode imputed smoking-related variables.
- Filled missing alcohol consumption values with 0.
- Removed BMI to avoid multicollinearity with height and weight.

All remaining missing values were resolved, and no duplicate rows were detected.

---

### 4.4 Distribution of Blood Pressure

#### 4.4.1 Distribution of Systolic and Diastolic Blood Pressure

This histogram shows the overall distribution of systolic and diastolic blood pressure across participants. Systolic BP exhibits a wider spread and slight right skew, whereas diastolic BP is more tightly clustered, indicating lower variability.

---

### 4.5 Relationships Between Blood Pressure and Key Features

#### 4.5.1 Age vs Systolic Blood Pressure

The scatter plot with a regression trendline illustrates a clear positive relationship between age and systolic blood pressure. As age increases, systolic BP tends to rise, consistent with clinical evidence identifying age as a major hypertension risk factor.

<img width="533" height="233" alt="image" src="https://github.com/user-attachments/assets/6af132d5-bae1-46a9-8005-dc9c14e30ee4" />

---

#### 4.5.2 Systolic Blood Pressure by Gender

This boxplot compares systolic BP between male and female participants. Males show a slightly higher median systolic BP and greater variability, suggesting gender-based differences in blood pressure distribution.

---

#### 4.5.3 Systolic Blood Pressure by Race/Ethnicity

This visualization highlights variations in systolic BP across different race and ethnicity groups. Some groups exhibit higher median BP and wider dispersion, indicating potential disparities influenced by genetic, socioeconomic, or lifestyle factors.

<img width="471" height="257" alt="image" src="https://github.com/user-attachments/assets/38af8eb3-8784-4924-8723-709dd5adb69b" />

---

#### 4.5.4 Impact of Smoking Status on Systolic Blood Pressure

This boxplot compares systolic BP among non-smokers, occasional smokers, and daily smokers. Daily smokers show a modestly higher BP distribution, suggesting smoking frequency may contribute to elevated blood pressure.

---

#### 4.5.5 Alcohol Consumption vs Systolic Blood Pressure

The scatter plot examines the association between average daily alcohol intake and systolic BP. The regression trendline indicates a mild positive relationship, suggesting higher alcohol consumption may be associated with increased blood pressure.

<img width="467" height="225" alt="image" src="https://github.com/user-attachments/assets/1681dbd7-37e5-4dcb-b8dc-b96c2b10905a" />

---

#### 4.5.6 Weight and Height vs Systolic Blood Pressure

This scatter plot visualizes systolic BP against body weight, with height encoded as a color gradient. Heavier individuals tend to show slightly higher systolic BP, although considerable variability exists, indicating interactions with other physiological factors.

<img width="568" height="241" alt="image" src="https://github.com/user-attachments/assets/20507d51-1e84-45fa-82f7-6f024afc1f35" />

---

### 4.6 EDA Summary

The EDA revealed meaningful relationships between systolic blood pressure and demographic and lifestyle variables such as age, gender, smoking behavior, alcohol intake, and body composition. These insights informed feature selection and validated the clinical relevance of the predictors used in subsequent modeling stages.

## 5. Model Training

This section describes the predictive modeling approach used to estimate systolic and diastolic blood pressure, including model selection, training strategy, evaluation metrics, and development environment.

---

### 5.1 Models Used for Predictive Analytics

To model blood pressure as a regression problem, the following algorithms were evaluated:

- **ElasticNet Regression**
  - Combines L1 (Lasso) and L2 (Ridge) regularization
  - Serves as a strong linear baseline and handles multicollinearity
- **Random Forest Regressor**
  - An ensemble, tree-based model capable of capturing nonlinear relationships
  - Robust to outliers and feature interactions
- **Gradient Boosting Regressor**
  - Sequential ensemble model that minimizes residual errors
  - Effective for structured tabular data

Each model was trained independently for:
- **Systolic Blood Pressure**
- **Diastolic Blood Pressure**

---

### 5.2 Feature and Target Definition

- **Target Variables**
  - `Systolic_BP`
  - `Diastolic_BP`

- **Feature Set**
  - All remaining demographic, lifestyle, and dietary features
  - Identifier columns (e.g., `Participant_ID`) and target variables were excluded from the feature matrix

---

### 5.3 Data Preprocessing Pipeline

A unified preprocessing pipeline was implemented using **scikit-learn Pipelines** to ensure consistency across all models.

**Numeric Features**
- Median imputation for missing values
- Standardization using `StandardScaler`

**Categorical Features**
- Mode imputation for missing values
- One-hot encoding with `handle_unknown='ignore'`

Both pipelines were combined using a `ColumnTransformer` and applied within each model pipeline to prevent data leakage.

---

### 5.4 Train–Test Split

The dataset was split into training and testing subsets using an **80/20 split**:

- **Training set**: 80%
- **Testing set**: 20%
- `random_state=42` was used to ensure reproducibility

---

### 5.5 Model Training and Hyperparameter Tuning

Each model was trained using **GridSearchCV** with **5-fold cross-validation**:

- **Cross-validation**: KFold (5 splits, shuffled)
- **Scoring metric**: R² (coefficient of determination)
- **Parallel processing**: Enabled using `n_jobs=-1`

Hyperparameters such as regularization strength, tree depth, number of estimators, and learning rate were tuned for optimal performance.

---

### 5.6 Evaluation Metrics

Model performance was evaluated using the following regression metrics:

- **Mean Absolute Error (MAE)** – average absolute prediction error
- **Root Mean Squared Error (RMSE)** – penalizes larger errors
- **R² Score** – proportion of variance explained by the model

These metrics were computed on the training set during cross-validation and used for model comparison.

---

### 5.7 Model Comparison Results

| Target Variable | Model | MAE | RMSE | R² |
|---------------|-------|-----|------|-----|
| Systolic_BP | ElasticNet | 11.21 | 15.12 | 0.305 |
| Systolic_BP | Random Forest | **8.84** | **12.16** | **0.550** |
| Systolic_BP | Gradient Boosting | 10.39 | 14.11 | 0.395 |
| Diastolic_BP | ElasticNet | 7.65 | 10.02 | 0.239 |
| Diastolic_BP | Random Forest | **5.69** | **7.53** | **0.570** |
| Diastolic_BP | Gradient Boosting | 6.89 | 9.04 | 0.380 |

---

### 5.8 Best Model Selection

Based on R² score and error metrics:
- **Random Forest Regressor** performed best for both systolic and diastolic blood pressure prediction.
- It consistently achieved lower MAE and RMSE while explaining the highest proportion of variance.

These best-performing models were retained for final evaluation on the test dataset.

---

### 5.9 Tools and Development Environment

- **Programming Language**: Python
- **Libraries**:
  - NumPy, Pandas
  - scikit-learn
- **Development Environment**:
  - Jupyter Notebook on local machine
- **Version Control**:
  - GitHub (for code and documentation)

---

## 6. Application of the Trained Models

To make the trained blood pressure prediction models accessible and interactive, a web application was developed using **Streamlit**. Streamlit was chosen due to its simplicity, rapid development capabilities, and seamless integration with Python-based machine learning workflows.

---

### 6.1 Web Application Overview

The Streamlit application allows users to input demographic, lifestyle, and dietary information and receive predicted:
- **Systolic Blood Pressure (SBP)**
- **Diastolic Blood Pressure (DBP)**

The application is designed for **early risk screening** and educational purposes, enabling users to understand how lifestyle factors may influence blood pressure levels.

---

### 6.2 Model Integration

- The **best-performing trained models** (Random Forest Regressors for both SBP and DBP) were serialized and loaded into the application.
- The same **preprocessing pipeline** used during model training (imputation, scaling, and encoding) is applied to user inputs to ensure consistency and prevent data leakage.
- Predictions are generated in real time based on user-provided inputs.

---

### 6.3 User Input Interface

The application collects the following inputs through intuitive UI components such as sliders, dropdowns, and numeric fields:
- Age
- Gender
- Race/Ethnicity
- Height and Weight
- Smoking Status
- Alcohol Consumption (Drinks per Day)
- Dietary Intake (Sodium, Potassium, Calories)

Input validation is applied to ensure reasonable and realistic values.

---

### 6.4 Output and Interpretation

- The predicted **systolic and diastolic blood pressure values** are displayed clearly on the results screen.
- Based on the predicted values, blood pressure categories (e.g., Normal, Elevated, Hypertension Stage 1/2) are shown to help users interpret the results.
- The interface emphasizes clarity and usability, making the predictions easy to understand for non-technical users.

---

### 6.5 Deployment and Environment

- **Framework**: Streamlit  
- **Programming Language**: Python  
- **Libraries Used**:
  - Streamlit
  - Pandas
  - NumPy
  - scikit-learn
- **Development Environment**:
  - Local Jupyter Notebook for model development
  - Streamlit Community Cloud for deployment

---

### 6.6 Limitations and Ethical Considerations

- The application is intended for **educational and screening purposes only** and does not replace professional medical advice.
- Predictions are based on historical NHANES data and may not generalize perfectly to all populations.
- Users are encouraged to consult healthcare professionals for clinical diagnosis and treatment.

---

## 7. Conclusion

### 7.1 Summary

This project successfully applied machine learning techniques to predict systolic and diastolic blood pressure using demographic, lifestyle, and dietary data derived from NHANES. The workflow included integrating multiple datasets, performing comprehensive exploratory data analysis, and building robust preprocessing pipelines. Several regression models were trained and tuned, with the Random Forest Regressor emerging as the best-performing algorithm for both target variables. Finally, the trained models were deployed through an interactive Streamlit web application, enabling real-time blood pressure prediction and user engagement.

---

### 7.2 Limitations

Despite promising results, the study has several limitations:
- Nutritional and lifestyle variables are largely self-reported, which may introduce reporting bias.
- NHANES data is cross-sectional in nature and does not support causal inference.
- Certain lifestyle variables exhibited high levels of missingness, requiring imputation or exclusion.
- Important contributors to blood pressure, such as stress levels, sleep quality, and genetic factors, were not available in the dataset.

These limitations should be considered when interpreting model predictions.

---

### 7.3 Future Work

Future enhancements could significantly strengthen the system:
- Incorporating additional NHANES cycles to increase sample size and population diversity.
- Adding model interpretability techniques such as SHAP values to explain individual predictions.
- Integrating external socioeconomic or environmental datasets (e.g., Census data).
- Deploying the application publicly to support broader real-world usage and feedback.

---

## 8. References

- National Health and Nutrition Examination Survey (NHANES) Data Documentation.  
  https://www.cdc.gov/nchs/nhanes/

- scikit-learn Documentation.  
  https://scikit-learn.org

- Streamlit Documentation.  
  https://streamlit.io
