# AI Skepticism Classification Project Plan

## Executive Summary
This project aims to build a machine learning model that predicts user skepticism levels towards AI responses based on various features including AI model characteristics, response attributes, and user demographics.

---

## Phase 1: Data Understanding & EDA

### 1.1 Dataset Structure Analysis
- **Target Variable**: `user_skepticism_category` (Classification)
  - Classes: Moderate Trust, Skeptical, Blind Trust, Highly Skeptical
  - This is a **multi-class classification** problem

### 1.2 Feature Categories

**AI Model Features:**
- `ai_model_name`: Claude, Llama, Mistral, ChatGPT-3.5, GPT-4, Gemini
- `ai_confidence_percentage`: Continuous (40-100%)
- `response_character_count`: Continuous

**Response Quality Features:**
- `has_cited_sources`: Boolean
- `contains_hedging_words`: Boolean
- `includes_disclaimer`: Boolean
- `answer_detail_level`: Vague → Very Specific (ordinal)
- `query_category`: 12 categories

**User Demographics:**
- `respondent_age_bracket`: 18-24, 25-34, 35-44, 45-54, 55-64, 65+
- `education_level`: High School, Bachelors, Masters, PhD, Professional
- `digital_literacy_score`: Low, Medium, High, Expert
- `ai_familiarity_level`: Beginner, First Time, Intermediate, Advanced, Expert

**Behavioral Features:**
- `decision_importance`: Low, Medium, Critical
- `urgency_level`: None, Low, Medium, High
- `performed_fact_check`: Boolean
- `fact_check_method_used`: 8 categories
- `verification_duration_mins`: Continuous
- `answer_accuracy_percentage`: Continuous
- `trust_calibration_valid`: Boolean

### 1.3 Key Data Quality Issues Identified
1. **Missing Values**: Expected in `fact_check_method_used` and `verification_duration_mins` (when `performed_fact_check` is FALSE)
2. **Class Imbalance**: Need to check distribution of skepticism categories
3. **Boolean Columns**: Stored as TRUE/FALSE strings, need conversion
4. **Categorical Encoding**: Multiple categorical variables need encoding

---

## Phase 2: Exploratory Data Analysis Plan

### 2.1 Statistical Summary
- Numerical features: mean, std, min, max, quartiles
- Categorical features: value counts, unique values

### 2.2 Distribution Analysis
- Histograms for numerical features
- Count plots for categorical features
- Box plots for outlier detection

### 2.3 Correlation Analysis
- Correlation matrix for numerical features
- Cramér's V for categorical-categorical relationships
- Point-biserial correlation for mixed types

### 2.4 Target Variable Analysis
- Class distribution (check for imbalance)
- Feature vs. target relationship plots
- Chi-square tests for independence

### 2.5 Outlier Detection
- IQR method for numerical features
- Z-score analysis
- Visual identification with boxplots

---

## Phase 3: Data Preprocessing Pipeline

### 3.1 Data Cleaning
1. Convert TRUE/FALSE strings to boolean/binary
2. Handle missing values:
   - For fact_check_method_used: Fill with "Not Performed" when not fact-checked
   - For verification_duration_mins: Fill with 0 when not fact-checked

### 3.2 Feature Engineering
1. **Binary Features Creation:**
   - `has_sources` (0/1)
   - `has_hedging` (0/1)
   - `has_disclaimer` (0/1)
   - `did_fact_check` (0/1)
   - `is_calibrated` (0/1)

2. **Ordinal Encoding:**
   - `answer_detail_level`: Vague=1 → Very Specific=4
   - `education_level`: High School=1 → PhD=4
   - `digital_literacy`: Low=1 → Expert=4
   - `ai_familiarity`: Beginner=1 → Expert=5

3. **New Features:**
   - `response_length_category`: Short, Medium, Long (based on percentiles)
   - `confidence_level`: Low, Medium, High (bins)
   - `is_high_stakes`: Critical decision importance

### 3.3 Categorical Encoding
- One-hot encoding for `ai_model_name`
- One-hot encoding for `query_category`
- One-hot encoding for `fact_check_method_used`

### 3.4 Feature Scaling
- StandardScaler for numerical features (ai_confidence_percentage, response_character_count, verification_duration_mins, answer_accuracy_percentage)

### 3.5 Train-Test Split
- 80-20 split
- Stratified sampling to maintain class distribution
- Random state = 42 for reproducibility

---

## Phase 4: Model Building Strategy

### 4.1 Baseline Models
1. **Logistic Regression** - Good interpretability baseline
2. **Dummy Classifier** - Majority class predictor

### 4.2 Tree-Based Models
1. **Random Forest** - Good for mixed feature types
2. **Gradient Boosting (XGBoost)** - High performance
3. **LightGBM** - Fast and efficient

### 4.3 Advanced Models
1. **Neural Network** (MLPClassifier) - If dataset size permits
2. **Support Vector Machine** - With RBF kernel

### 4.4 Cross-Validation Strategy
- 5-fold Stratified K-Fold CV
- Evaluation metrics: Accuracy, F1-Score (macro), ROC-AUC (macro)

---

## Phase 5: Model Evaluation Framework

### 5.1 Classification Metrics
- **Accuracy**: Overall correctness
- **F1-Score**: Precision-Recall balance (important for imbalanced classes)
- **ROC-AUC**: Probability calibration quality
- **Classification Report**: Per-class metrics
- **Confusion Matrix**: Visual error analysis

### 5.2 Advanced Diagnostics
- **Learning Curves**: Detect overfitting/underfitting
- **Validation Curves**: Hyperparameter sensitivity
- **Shap Values**: Feature importance explanation

### 5.3 Model Comparison
- Cross-validation scores comparison table
- Statistical significance testing (McNemar's test)

---

## Phase 6: Hyperparameter Tuning

### 6.1 Random Forest Tuning
```python
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### 6.2 XGBoost Tuning
```python
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

### 6.3 Tuning Strategy
- RandomizedSearchCV for initial exploration (n_iter=50)
- GridSearchCV for final optimization (refined grid)
- 5-fold cross-validation throughout

---

## Phase 7: Final Model Selection

### 7.1 Selection Criteria
1. Best F1-Score (macro) on validation set
2. Stability across cross-validation folds
3. Interpretability requirements
4. Inference speed requirements

### 7.2 Feature Importance Analysis
- Mean decrease in impurity (MDI)
- Permutation importance
- SHAP values for global and local explanations

### 7.3 Model Persistence
- Save best model using joblib/pickle
- Create preprocessing pipeline with joblib

---

## Deployment Preparation Phase 8:

### 8.1 API Development
- Flask REST FastAPI alternative API endpoint
- option
- Request/response schema documentation

### 8.2 Web Application
- Streamlit dashboard for predictions
- Feature importance visualization
- Prediction confidence display

### 8.3 Code Structure
```
ml_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── trained/
│   └── scalers/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── notebooks/
├── api/
├── app/
├── requirements.txt
└── README.md
```

---

## Phase 9: Documentation & Reporting

### 9.1 Technical Documentation
- Data dictionary
- Preprocessing pipeline explanation
- Model card (performance metrics, limitations)
- Feature importance report

### 9.2 Business Insights
- Key factors driving skepticism
- Recommendations for AI system improvement
- Risk assessment based on model predictions

---

## Implementation Steps

1. ✅ Dataset loaded and analyzed
2. ⬜ Create EDA notebook with visualizations
3. ⬜ Build preprocessing pipeline
4. ⬜ Train and evaluate baseline models
5. ⬜ Implement advanced models with cross-validation
6. ⬜ Perform hyperparameter tuning
7. ⬜ Select final model and analyze feature importance
8. ⬜ Create deployment-ready API and Streamlit app
9. ⬜ Document findings and insights

---

## Dependencies Required
```
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
plotly
joblib
flask
fastapi
streamlit
shap
jupyter
```

---

## Expected Outcomes
1. Production-ready ML model with >85% accuracy
2. Complete preprocessing pipeline
3. Deployed prediction API
4. Interactive Streamlit dashboard
5. Comprehensive documentation
6. Business recommendations for reducing AI skepticism

---

*Project Plan Version: 1.0*
*Created: 2024*
*Last Updated: [Current Date]*

