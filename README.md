# 💳 Credit Card Fraud Detection

## 📌 Overview
This project builds a Machine Learning model to detect fraudulent credit card transactions using Logistic Regression.  
The dataset is highly imbalanced, so evaluation is done using both accuracy and classification metrics such as precision, recall, and F1-score.

---

## 📂 Project Structure

```
credit-card-fraud-ml/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── fraud_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## 🚀 How to Run

### 1. Clone the Repository
```
git clone https://github.com/your-username/credit-card-fraud-ml.git
cd credit-card-fraud-ml
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Model
```
python -m src.train
```

---

## 📊 Output

Example output:

```
✅ Data Loaded Successfully
Shape: (284807, 31)
Missing values: 0

📊 Model Performance
Accuracy score on Test Data : 0.9238578680203046

📄 Classification Report:

              precision    recall  f1-score   support
           0       0.99      0.99      0.99     56864
           1       0.85      0.60      0.70        98

    accuracy                           0.99     56962
   macro avg       0.92      0.79      0.85     56962
weighted avg       0.99      0.99      0.99     56962
```

---

## 🧠 Key Features
- Data preprocessing pipeline
- Logistic Regression model
- Stratified train-test split for imbalanced data
- Accuracy and classification report evaluation
- Model saving using pickle

---

## ⚠️ Dataset Info
- Total Features: 30  
- Target Column: `Class`  
- `0` → Legit Transaction  
- `1` → Fraud Transaction  

Note: Most features (V1–V28) are PCA-transformed and not directly interpretable.

---

## 📈 Future Improvements
- Implement advanced models (Random Forest, XGBoost)
- Handle class imbalance using SMOTE
- Add confusion matrix and ROC curve visualization
- Convert into REST API (Flask/FastAPI)
- Deploy the model online

---

## 👨‍💻 Author
**Lakshya Prasad**  
B.Tech CSE (AI)  
Data Analytics | Machine Learning