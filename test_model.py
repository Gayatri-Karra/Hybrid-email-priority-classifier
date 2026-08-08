import joblib

# Load saved model
model = joblib.load('model/email_priority_pipeline.pkl')

# Test prediction
message = 'Urgent: database server failed'

prediction = model.predict([message])[0]

print('Message:', message)
print('Predicted Priority:', prediction)

print('Model loaded successfully!')