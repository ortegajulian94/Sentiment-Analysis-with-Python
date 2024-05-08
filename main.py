# main.py

import tarfile
import os
from utils import load_data
from preprocessing import preprocess_text
from model import train_model, evaluate_model, predict_sentiment
from sklearn.model_selection import train_test_split

# Extract IMDb Movie Reviews dataset
print("Extracting IMDb Movie Reviews dataset...")
try:
    with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
        tar.extractall()
    print("Dataset extracted successfully.")
except Exception as e:
    print("Error extracting dataset:", e)


# Load data
print("Loading data...")
try:
    data = load_data("aclImdb")
    print("Data loaded successfully.")
except Exception as e:
    print("Error loading data:", e)
    raise  # Re-raise the exception to halt execution

# Preprocess the data
print("Preprocessing data...")
try:
    data['preprocessed_text'] = data['text'].apply(preprocess_text)
    print("Data preprocessed successfully.")
except Exception as e:
    print("Error preprocessing data:", e)
    raise  # Re-raise the exception to halt execution

# Split the dataset into features (X) and labels (y)
print("Splitting dataset into features and labels...")
X = data['preprocessed_text']
y = data['sentiment']

# Proceed with the remaining steps...



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset split into training and testing sets.")

# Train the model
print("Training the model...")
model = train_model(X_train, y_train)

print("Model trained successfully.")

# Evaluate the model
print("Evaluating the model...")
evaluation_metrics = evaluate_model(model, X_test, y_test)
print("Evaluation Metrics:")
print(evaluation_metrics)

# Additional code for task:

# Load a subset of the IMDb Movie Reviews dataset
data_subset = data.head(5)  # Load first 5 rows for demonstration

# Extract text samples from the loaded dataset
sample_text = data_subset['text'].tolist()

# Predict sentiment labels for the sample text using the trained model
print("Performing prediction task...")
predictions = predict_sentiment(model, sample_text)

print("Prediction task completed successfully.")

# Display the predicted sentiment labels
print("Predicted sentiment labels for sample text:")
for text, sentiment in zip(sample_text, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
