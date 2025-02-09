import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Streamlit UI
st.title("Credit Card Fraud Detection System")

# File Upload Section
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

def load_data(file_path=None):
    if file_path:
        return pd.read_csv(file_path)
    return pd.read_csv("creditcard.csv")  # Default file

# Load dataset
data = load_data(uploaded_file) if uploaded_file else load_data()

# Button to show/hide dataset
view_dataset = st.sidebar.checkbox("View Dataset")
selected_index = None
if view_dataset:
    st.subheader("📂 Full Dataset")
    selected_index = st.number_input("Select a row index", min_value=0, max_value=len(data)-1, step=1, value=0)
    st.dataframe(data)

# Automatically fill input with selected row but do not trigger enter key behavior
def get_default_input():
    if selected_index is not None:
        selected_row = data.iloc[selected_index, :-1].values  # Exclude 'Class' column
        return ", ".join(map(str, selected_row))
    return ""

default_input = get_default_input()

# Balance dataset using undersampling
def balance_data(df):
    legit_transactions = df[df.Class == 0]
    fraud_transactions = df[df.Class == 1]
    legit_sample = legit_transactions.sample(n=len(fraud_transactions), random_state=42)
    balanced_df = pd.concat([legit_sample, fraud_transactions], axis=0).sample(frac=1, random_state=42)
    return balanced_df

data = balance_data(data)

# Split data into features and labels
X = data.drop(columns=["Class"], axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
train_accuracy = accuracy_score(model.predict(X_train), y_train)
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# User instructions
st.write("Select a row from the dataset, then press 'Check Transaction' to evaluate.")

# User input field with row selection auto-fill
user_input = st.text_area("Enter feature values separated by commas:", value=default_input, key="user_input")

if st.button("Check Transaction"):
    try:
        input_features = np.array([float(i) for i in st.session_state.user_input.split(',')], dtype=np.float64)
        prediction = model.predict(input_features.reshape(1, -1))
        result_message = "✅ Transaction is Legitimate" if prediction[0] == 0 else "🚨 Transaction is Fraudulent"
        st.success(result_message)
    except ValueError:
        st.error("Invalid input! Please enter numerical values separated by commas.")

# Display model accuracy and additional metrics
st.sidebar.header("Model Performance")
st.sidebar.write(f"Training Accuracy: {train_accuracy:.2f}")
st.sidebar.write(f"Testing Accuracy: {test_accuracy:.2f}")
st.sidebar.write(f"Precision: {precision:.2f}")
st.sidebar.write(f"Recall: {recall:.2f}")
st.sidebar.write(f"F1 Score: {f1:.2f}")
