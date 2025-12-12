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
- **YouTube Video Presentation:** *To be added*    

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

A detailed exploratory data analysis was conducted to understand the distribution of blood pressure values, identify potential predictors, and inspect relationships between lifestyle variables and the target variables.

### 4.1 Summary Statistics
Basic descriptive statistics were computed for all numeric variables, including blood pressure, age, height, weight, sodium, potassium, and calorie intake.

Key observations:
- Systolic BP ranged from 70 to 232 mmHg, with a mean of ~119 mmHg.
- Diastolic BP ranged from 34 to 139 mmHg, with a mean of ~72 mmHg.
- Dietary variables (sodium, potassium, calories) showed high variance.
- Height and weight distributions aligned with typical NHANES population distributions.

### 4.2 Missing Data Analysis

| Column | % Missing |
|--------|-----------|
| Cigarettes_Per_Day | 97% |
| Current_Smoking_Status | 67% |
| Drinks_Per_Day | 46% |
| Sodium_mg / Potassium_mg / Calories_kcal | 22% |
| BMI / Weight / Height | <1% |

Steps taken:
- Dropped `Cigarettes_Per_Day`.
- Median imputed sodium, potassium, and calorie intake.
- Mode imputed smoking variables; filled missing drinking values with 0.
- Median imputed height and weight.
- Removed BMI to avoid multicollinearity.

All remaining missing values were resolved.

### 4.3 Visual Insights

Key EDA findings:
- Systolic BP increases linearly with age.
- Males typically have higher systolic BP than females.
- Meaningful variation across race/ethnicity groups.
- Daily smokers show higher systolic BP than non-smokers.
- Mild positive trend between alcohol intake and BP levels.
- Heavier individuals show modest increases in systolic BP.

These insights guided model selection and feature engineering.

---

## 5. Model Training

### 5.1 Modeling Approach
Two regression targets were modeled independently:
- Systolic_BP
- Diastolic_BP

A preprocessing pipeline handled numeric scaling and categorical encoding.

### 5.2 Train–Test Split
- Training: 80% (6014 samples)
- Testing: 20% (1504 samples)

### 5.3 Preprocessing Pipelines

**Numeric pipeline**
- Median imputation
- Standard scaling

**Categorical pipeline**
- Mode imputation
- One-hot encoding

These pipelines were integrated using `ColumnTransformer`.

### 5.4 Model Candidates

| Model | Description |
|-------|-------------|
| ElasticNet Regression | Linear + regularization |
| Random Forest Regressor | Nonlinear, robust |
| Gradient Boosting Regressor | Sequential tree boosting |

Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation.

### 5.5 Evaluation Metrics
Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 5.6 Results

| Target | Best Model | MAE | RMSE | R² |
|--------|------------|-----|------|-----|
| Systolic BP | Random Forest | 8.84 | 12.16 | 0.55 |
| Diastolic BP | Random Forest | 5.69 | 7.53 | 0.57 |

Random Forest outperformed all other models on both targets.

### 5.7 Model Saving
The trained models were exported as:

rf_systolic.pkl
rf_diastolic.pkl


These files are loaded by the Streamlit app for inference.

---

## 6. Application of the Trained Models

A Streamlit application was developed to make the model interactive and accessible.

### 6.1 Features of the App
- User inputs demographic and lifestyle variables.
- Predicts systolic and diastolic BP in real time.
- Provides a simple interface for non-technical users.
- Loads pre-trained model files for fast inference.

### 6.2 Deployment
The app is run locally with:
streamlit run simple_bp_app.py


It can be deployed to Streamlit Cloud, AWS, or Azure for public access.

---

## 7. Conclusion

### Summary
The project:
- Integrated multiple NHANES datasets.
- Conducted thorough exploratory data analysis.
- Built and tuned multiple machine-learning models.
- Identified Random Forest as the best-performing algorithm.
- Created a functional Streamlit application for prediction.

### Limitations
- Self-reported nutritional data may contain bias.
- NHANES data is cross-sectional; does not support causal inference.
- Some lifestyle variables had high missingness.
- Important factors like stress and sleep were not available.

### Future Work
- Incorporate more NHANES cycles for larger sample size.
- Add interpretability features (e.g., SHAP values).
- Expand the app to include BP category classification.
- Integrate external environmental and socioeconomic data.
- Deploy the app publicly for real-world usage.

---

## 8. References

- NHANES Data Documentation – https://www.cdc.gov/nchs/nhanes/
- scikit-learn Documentation – https://scikit-learn.org
- Streamlit Documentation – https://streamlit.io
- Internal code, analysis, and data-processing steps from project notebooks
