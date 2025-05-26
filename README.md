# 🕵️‍♂️ Fake Profile Detection on Social Media using Machine Learning

This project aims to identify and classify fake user profiles on social media platforms using machine learning techniques. It uses a combination of feature engineering, natural language processing, and classification models to distinguish between genuine and fake accounts.

---

## 📌 Project Overview

- **Goal:** Detect fake profiles using publicly available or simulated social media data.
- **Approach:** Supervised machine learning using engineered features from user metadata and textual analysis.
- **Outcome:** Achieved high accuracy in classifying profiles as fake or real using multiple ML models.

---

## 🔍 Features Used

- Follower-to-following ratio  
- Username patterns and length  
- Presence of profile picture and bio  
- Frequency of posts and likes  
- NLP-based features from bio and posts (e.g., TF-IDF, sentiment, keyword flags)

---

## 🧠 Machine Learning Models Used

- XGBoost  

---

## 📊 Performance Metrics

- Accuracy  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC-AUC Score

---

## 🛠️ Technologies Used

- **Python**
- **Scikit-learn** – ML models & preprocessing  
- **Pandas, NumPy** – data manipulation  
- **Matplotlib, Seaborn** – visualization  
- **NLTK / spaCy** – text processing  
- **XGBoost** – gradient boosting  
- **SMOTE** – to address class imbalance  
- **Streamlit / Flask** – for model demo deployment  
- **Git / GitHub** – version control

---

## 🚀 Deployment

A prototype interface was created using **Streamlit** to input and analyze user profile data and return predictions in real-time.


---

## 📂 Project Structure

```
├── data/                 # Raw and processed datasets
├── models/               # Trained ML models (Pickle files)
├── notebooks/            # Jupyter notebooks for EDA & training
├── app.py                # Streamlit/Flask web app
├── utils.py              # Utility functions
├── requirements.txt      # Required packages
└── README.md             # Project documentation
```

---


## 📚 Future Work

- Improve feature extraction using advanced NLP models (BERT, GPT-based embeddings)
- Integrate real-time data collection APIs
- Extend to multi-class classification (bot, spam, inactive, etc.)
- Model deployment using Docker and AWS/GCP

---

## 👨‍💻 Author

**Your Name**  
Email: chandrikachavva5@gmail.com  
LinkedIn: ( https://www.linkedin.com/in/chandrika508/)
GitHub: (https://github.com/chandrika203)

---

