import streamlit as st
import joblib
import pandas as pd
import re

# Load model and vectorizer
model = joblib.load('model/priority_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Page settings
st.set_page_config(
    page_title='Hybrid Email Priority Classifier',
    page_icon='📧',
    layout='centered'
)

# Keywords
HIGH_KEYWORDS = ['urgent', 'immediately', 'asap', 'critical', 'server down', 'payment failed', 'security breach']
PROMO_WORDS = [
    'sale', 'offer', 'discount', 'buy now', 'coupon',
    'deal', 'flash sale', 'limited time', 'shop now'
]
LOW_KEYWORDS = ['thanks', 'thank you', 'lunch', 'coffee', 'birthday', 'movie', 'weekend']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def predict_priority(message):
    text = clean_text(message)
    if any(p in text for p in PROMO_WORDS):
        return 'Low', 0.99, 'Promotional safeguard'
    for kw in HIGH_KEYWORDS:
        if kw in text:
            return 'High', 0.99, f'Rule-based keyword: {kw}'

    for kw in LOW_KEYWORDS:
        if kw in text:
            return 'Low', 0.99, f'Rule-based keyword: {kw}'

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = max(probs)

    return pred, confidence, 'ML model'

# UI
st.title('📧 Hybrid Email / Message Priority Classifier')

st.markdown(
    'This app combines **rule-based NLP** with **machine learning (TF-IDF + Logistic Regression)**.'
)

st.subheader('Try an example')

col1, col2, col3 = st.columns(3)

if col1.button('🚨 High'):
    st.session_state.example = 'Urgent: production server is down'

if col2.button('⚠️ Medium'):
    st.session_state.example = 'Please review the project report tomorrow'

if col3.button('✅ Low'):
    st.session_state.example = 'Thanks for lunch today'

default_text = st.session_state.get('example', '')

user_input = st.text_area(
    'Enter email or message',
    value=default_text,
    height=140
)

if st.button('Predict Priority'):

    if user_input.strip() == '':
        st.warning('Please enter a message.')
    else:
        prediction, confidence, source = predict_priority(user_input)

        if prediction == 'High':
            st.error(f'🚨 Predicted Priority: {prediction}')
        elif prediction == 'Medium':
            st.warning(f'⚠️ Predicted Priority: {prediction}')
        else:
            st.success(f'✅ Predicted Priority: {prediction}')

        st.metric('Confidence', f'{confidence*100:.1f}%')

        st.info(f'Prediction source: {source}')

        st.subheader('Entered Message')
        st.write(user_input)

        report = pd.DataFrame({
            'Message': [user_input],
            'Prediction': [prediction],
            'Confidence': [f'{confidence*100:.1f}%'],
            'Source': [source]
        })

        csv = report.to_csv(index=False).encode('utf-8')

        st.download_button(
            label='📥 Download Prediction Report',
            data=csv,
            file_name='prediction_report.csv',
            mime='text/csv'
        )

st.sidebar.title('About')

st.sidebar.markdown('### Tech Stack')
st.sidebar.markdown('- Python')
st.sidebar.markdown('- scikit-learn')
st.sidebar.markdown('- TF-IDF')
st.sidebar.markdown('- Logistic Regression')
st.sidebar.markdown('- Streamlit')

st.sidebar.markdown('### Model')
st.sidebar.write('Hybrid NLP + ML classifier for email priority prediction.')
#Batch Email Analyzer

st.markdown('---')
st.header('📥 Batch Email Analyzer')

uploaded_file = st.file_uploader(
    'Upload emails.csv',
    type=['csv']
)

if uploaded_file is not None:
    emails_df = pd.read_csv(uploaded_file)

    results = []

    for _, row in emails_df.iterrows():
        combined_text = f"{row['subject']} {row['body']}"
        pred, conf, source = predict_priority(combined_text)

        results.append({
            'Sender': row['sender'],
            'Subject': row['subject'],
            'Priority': pred,
            'Confidence': f'{conf*100:.1f}%',
            'Source': source
        })

    result_df = pd.DataFrame(results)

    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    result_df['sort_key'] = result_df['Priority'].map(priority_order)
    result_df = result_df.sort_values('sort_key').drop(columns=['sort_key'])

    st.subheader('📌 Prioritized Inbox')
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label='📥 Download Prioritized Inbox',
        data=csv,
        file_name='prioritized_inbox.csv',
        mime='text/csv'
    )