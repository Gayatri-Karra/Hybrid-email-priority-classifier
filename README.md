# Hybrid Email / Message Priority Classifier

A practical NLP application that classifies emails and messages into **High, Medium, or Low priority** using a **hybrid architecture**:

* **Rule-based NLP** for clearly urgent or casual messages
* **Machine Learning (TF-IDF + Logistic Regression)** for ambiguous messages

The application is deployed with **Streamlit** and provides real-time predictions, confidence scores, and downloadable prediction reports.

---

## Demo Features

* Predicts **High / Medium / Low** priority
* Displays **confidence score**
* Shows **prediction source** (Rule-based or ML)
* Downloadable CSV prediction report
* Simple and interactive Streamlit web interface

---

## Tech Stack

* Python
* pandas
* scikit-learn
* Streamlit
* Joblib

---

## Project Structure

```text
email_priority_classifier/
├── data/
│   └── messages.csv
├── model/
│   ├── priority_model.pkl
│   └── tfidf_vectorizer.pkl
├── train_model.py
├── test_model.py
├── app.py
└── README.md
```

---

## Machine Learning Workflow

1. Collect and label messages
2. Preprocess text (lowercase, remove punctuation)
3. Convert text to numerical features using **TF-IDF**
4. Train a **Logistic Regression** classifier
5. Evaluate model performance
6. Save the trained model using **Joblib**
7. Deploy the application with **Streamlit**

---

## Sample Predictions

| Message                                     | Prediction |
| ------------------------------------------- | ---------- |
| `Urgent: server is down`                    | High       |
| `Please review the project report tomorrow` | Medium     |
| `Thanks for lunch today`                    | Low        |

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn streamlit joblib
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Run the web application

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Example Output

```text
Message: Please review the project report tomorrow
Prediction: Medium
Confidence: 76.3%
Source: ML model
```

---

## Problem Statement

Organizations receive a large number of emails and messages every day. Manually identifying which messages require immediate attention is time-consuming. This project automatically classifies incoming messages into **High, Medium, or Low priority** categories so that important communications can be handled first.

---

## Future Improvements

* Larger real-world email dataset
* Transformer-based NLP models (BERT)
* Email subject + body classification
* User feedback loop for continuous learning
* Deployment on Streamlit Cloud

---

## Resume Highlight

**Hybrid Email / Message Priority Classifier using TF-IDF, Logistic Regression, and Streamlit**

Built a hybrid NLP application that combines rule-based filtering with machine learning to classify messages by priority and provide confidence-aware predictions through a web interface.
