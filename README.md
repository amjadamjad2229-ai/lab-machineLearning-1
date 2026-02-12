# 🤖 AI Skepticism Classification Project

A complete machine learning project for predicting user skepticism levels towards AI responses. This project includes data analysis, model training, hyperparameter tuning, and deployment-ready code.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Details](#pipeline-details)
- [Model Performance](#model-performance)
- [API Usage](#api-usage)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Feature Importance](#feature-importance)
- [Key Findings](#key-findings)
- [Deployment](#deployment)

---

## 🎯 Project Overview

This project builds a machine learning model that predicts **user skepticism categories** towards AI responses based on:

1. **AI Model Characteristics**: Model name, confidence level
2. **Response Quality**: Source citation, hedging words, disclaimers, detail level
3. **User Demographics**: Age, education, digital literacy, AI familiarity
4. **Behavioral Factors**: Decision importance, urgency, fact-checking behavior

### Problem Type
- **Multi-class Classification** with 4 target classes:
  - 🟢 **Moderate Trust**: Users with balanced trust in AI
  - 🔵 **Blind Trust**: Users who trust AI without verification
  - 🔴 **Highly Skeptical**: Users with very low trust
  - 🟡 **Skeptical**: Users with moderate skepticism

---

## 📊 Dataset Description

| Feature | Type | Description |
|---------|------|-------------|
| `ai_model_name` | Categorical | Claude, GPT-4, ChatGPT-3.5, Gemini, Llama, Mistral |
| `query_category` | Categorical | 12 query types (medical, legal, financial, etc.) |
| `ai_confidence_percentage` | Numerical | AI's self-reported confidence (0-100) |
| `response_character_count` | Numerical | Length of AI response |
| `has_cited_sources` | Boolean | Whether AI cited sources |
| `contains_hedging_words` | Boolean | Use of hedging language |
| `includes_disclaimer` | Boolean | Presence of disclaimer |
| `answer_detail_level` | Ordinal | Vague → Very Specific |
| `respondent_age_bracket` | Categorical | 6 age groups |
| `education_level` | Ordinal | High School → PhD |
| `digital_literacy_score` | Ordinal | Low → Expert |
| `ai_familiarity_level` | Ordinal | Beginner → Expert |
| `decision_importance` | Ordinal | None → Critical |
| `urgency_level` | Ordinal | None → High |
| `performed_fact_check` | Boolean | User verification behavior |
| `answer_accuracy_percentage` | Numerical | Actual answer accuracy |
| `trust_score_out_of_10` | Numerical | User trust rating |
| **Target: `user_skepticism_category`** | Categorical | 4 skepticism levels |

---

## 📁 Project Structure

```
ai_skepticism_project/
├── ai_skepticism_dataset.csv      # Raw dataset
├── ai_skepticism_ml_pipeline.py   # Main ML pipeline
├── api.py                         # FastAPI application
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md                     # This file
│
├── models/
│   └── best_model.pkl            # Trained model (after running pipeline)
│
├── plots/                         # EDA visualizations (generated)
│   ├── target_distribution.png
│   ├── ai_model_distribution.png
│   ├── confidence_by_skepticism.png
│   ├── response_length_distribution.png
│   ├── correlation_matrix.png
│   ├── skepticism_by_age.png
│   ├── skepticism_by_ai_model.png
│   └── feature_importance.png
│
└── notebooks/
    └── eda_notebook.ipynb        # Jupyter notebook for exploration
```

---

## 🛠️ Installation

### 1. Clone and Install Dependencies

```bash
cd machinelearinglabs
pip install -r requirements.txt
```

### 2. Install Optional Dependencies

```bash
# For Jupyter notebooks
pip install jupyter notebook

# For feature importance visualization
pip install shap
```

---

## 🚀 Quick Start

### Option 1: Run Full Pipeline

```bash
python ai_skepticism_ml_pipeline.py
```

This will:
- Load and explore the dataset
- Generate EDA visualizations in `plots/`
- Train 7 different models
- Identify the best performing model
- Save the model to `models/best_model.pkl`

### Option 2: Run Individual Components

```python
# Just load and explore data
from ai_skepticism_ml_pipeline import load_data, basic_statistics, assess_data_quality
df = load_data('ai_skepticism_dataset.csv')

# Just train models
from ai_skepticism_ml_pipeline import train_baseline_models
results = train_baseline_models(X_train, y_train, X_test, y_test)
```

---

## 🔬 Pipeline Details

### Step 1: Data Quality Assessment

The pipeline performs comprehensive data quality checks:

```python
# Missing value analysis
# Boolean conversion (TRUE/FALSE → 1/0)
# Target class distribution analysis
# Outlier detection
```

### Step 2: Feature Engineering

New features created during preprocessing:

| Feature | Description |
|---------|-------------|
| `is_high_stakes` | Decision importance = Critical |
| `is_expert_user` | Digital literacy = Expert |
| `is_high_confidence_ai` | AI confidence > 80% |
| `is_detailed_response` | Response length > median |

Ordinal encoding for:
- `answer_detail_level` (1-4)
- `education_level` (1-5)
- `digital_literacy_score` (1-4)
- `ai_familiarity_level` (1-5)

### Step 3: Model Training

7 models trained with 5-fold stratified cross-validation:

| Model | Strengths |
|-------|-----------|
| Logistic Regression | Interpretability, baseline |
| Random Forest | Robust to outliers, feature importance |
| Gradient Boosting | Good generalization |
| XGBoost | High performance, regularization |
| LightGBM | Fast training, large datasets |
| Neural Network | Non-linear patterns |
| Dummy Classifier | Baseline comparison |

### Step 4: Evaluation Metrics

- **Accuracy**: Overall correctness
- **F1-Score (Macro)**: Balance between precision and recall
- **ROC-AUC**: Probability calibration
- **Confusion Matrix**: Error analysis
- **Learning Curves**: Overfitting detection

---

## 📈 Model Performance

### Expected Results (after running pipeline)

| Metric | Value | Notes |
|--------|-------|-------|
| Best Model | XGBoost / LightGBM | Typically outperforms others |
| Test Accuracy | ~85-90% | Depends on data split |
| F1-Score (Macro) | ~0.82-0.87 | Balanced across classes |
| Cross-Val F1 | ~0.80-0.85 | Stable performance |

### Feature Importance Insights

The most influential features typically include:

1. **Trust Score** - Strong predictor of skepticism
2. **Answer Accuracy** - Higher accuracy → more trust
3. **AI Confidence** - Self-reported vs actual
4. **Decision Importance** - High stakes → more skepticism
5. **Digital Literacy** - Expert users → more skeptical

---

## 🌐 API Usage

### Start the API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Health Check
```bash
GET /
```
Response:
```json
{
  "status": "healthy",
  "message": "AI Skepticism Prediction API is running",
  "model_loaded": true
}
```

#### Make Prediction
```bash
POST /predict
```

Request body:
```json
{
  "ai_model_name": "Claude",
  "query_category": "medical_advice",
  "ai_confidence_percentage": 75.5,
  "response_character_count": 500,
  "has_cited_sources": true,
  "contains_hedging_words": false,
  "includes_disclaimer": true,
  "answer_detail_level": "Specific",
  "respondent_age_bracket": "35-44",
  "education_level": "Bachelors",
  "digital_literacy_score": "Medium",
  "ai_familiarity_level": "Intermediate",
  "decision_importance": "High",
  "urgency_level": "Medium",
  "performed_fact_check": true,
  "answer_accuracy_percentage": 85.0,
  "trust_calibration_valid": true
}
```

Response:
```json
{
  "prediction": "Moderate Trust",
  "confidence": 0.87,
  "all_probabilities": {
    "Blind Trust": 0.05,
    "Highly Skeptical": 0.08,
    "Moderate Trust": 0.87,
    "Skeptical": 0.00
  },
  "model_version": "1.0.0"
}
```

#### Batch Prediction
```bash
POST /batch-predict
```
Process multiple predictions at once.

---

## 📊 Streamlit Dashboard

### Start Dashboard

```bash
streamlit run app.py
```

### Dashboard Features

1. **🔮 Prediction Tab**
   - Interactive input form
   - Real-time predictions
   - Probability distribution visualization

2. **📊 Feature Importance Tab**
   - Horizontal bar chart of feature importance
   - Interactive filtering
   - Key insights

3. **📈 Model Info Tab**
   - Model architecture details
   - All features list
   - Target class distribution

---

## 🔍 Key Findings

### Skepticism Patterns

Based on EDA and model analysis:

#### 1. **Age & Skepticism**
- Older users (65+) tend to be more skeptical
- Younger users (18-24) show higher blind trust
- Middle-aged users (35-54) have moderate trust

#### 2. **Education Impact**
- Higher education → more skepticism (critical thinking)
- PhD holders are often more skeptical
- High School education correlates with blind trust

#### 3. **AI Model Differences**
- Newer models (GPT-4, Claude) inspire more trust
- Open-source models (Llama) show varied trust patterns
- Model performance correlates with trust levels

#### 4. **Query Type Matters**
- **High skepticism areas**: Legal advice, medical advice, financial advice
- **Low skepticism areas**: Recipe cooking, creative writing, general knowledge
- Critical decisions → higher skepticism

#### 5. **Response Characteristics**
- Detailed responses → more trust
- Cited sources → more trust
- Hedging words → more skepticism
- Disclaimers → mixed effect

### Business Recommendations

1. **For AI Developers**:
   - Provide detailed responses with citations
   - Be transparent about limitations
   - Include confidence levels

2. **For Users**:
   - Critical decisions should involve fact-checking
   - Don't ignore disclaimers
   - Match AI capabilities to task complexity

3. **For Organizations**:
   - Train users on AI limitations
   - Implement verification workflows
   - Use appropriate AI for task criticality

---

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment Options

| Platform | Instructions |
|----------|--------------|
| **Heroku** | Connect GitHub repo, set buildpack |
| **AWS** | Use Elastic Beanstalk or ECS |
| **GCP** | Use Cloud Run |
| **Azure** | Use App Service |

### Environment Variables

```bash
export MODEL_PATH=models/best_model.pkl
export API_HOST=0.0.0.0
export API_PORT=8000
```

---

## 📝 Notes

### Reproducibility

- Set `RANDOM_STATE = 42` for all random operations
- Use stratified sampling for train-test split
- Cross-validation with fixed random state

### Limitations

1. **Dataset Size**: ~1500 samples may limit model complexity
2. **Class Imbalance**: May need SMOTE or class weights
3. **Feature Availability**: Some features may not be available in production
4. **Temporal Effects**: Skepticism may change over time

### Future Improvements

1. Collect more data for rare skepticism classes
2. Implement online learning for model updates
3. Add more AI model features
4. Include user interaction history
5. Implement A/B testing framework

---

## 📧 Contact

For questions or contributions, please open an issue or contact the project maintainer.

---

## 🙏 Acknowledgments

- Dataset from Kaggle AI Skepticism research
- Built with scikit-learn, XGBoost, LightGBM
- Visualization with matplotlib and seaborn
- Deployment with FastAPI and Streamlit

---

**Made with ❤️ for the ML Community**

