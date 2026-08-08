import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('data/messages.csv')

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['priority']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f'Test Accuracy: {accuracy:.2f}')
print('\\nClassification Report:\\n')
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/priority_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')

print('\\nModel saved successfully!')