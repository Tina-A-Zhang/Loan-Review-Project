import os
import pandas as pd
import tensorflow as tf  # or torch, scikit-learn
import joblib
from bronze import bronze_layer_transformation_chunked
from silver import silver_layer_transformation_chunked
from gold import gold_layer_transformation_chunked
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define file paths
RAW_DATA_PATH = os.path.join('data', 'raw_data.csv')
BRONZE_DATA_PATH = os.path.join('data', 'bronze_output.parquet')
SILVER_DATA_PATH = os.path.join('data', 'silver_output.parquet')
GOLD_DATA_PATH = os.path.join('data', 'gold_output.parquet')
MODEL_PATH = 'trained_model.h5'

def run_full_pipeline():
    """
    Orchestrates the entire data cleaning, feature engineering, and model training pipeline.
    """
    print("--- Starting Bronze Layer Transformation ---")
    bronze_layer_transformation_chunked(RAW_DATA_PATH, BRONZE_DATA_PATH)
    print("--- Bronze Layer Complete ---")

    print("--- Starting Silver Layer Transformation ---")
    silver_layer_transformation_chunked(BRONZE_DATA_PATH, SILVER_DATA_PATH)
    print("--- Silver Layer Complete ---")

    print("--- Starting Gold Layer Transformation ---")
    gold_layer_transformation_chunked(SILVER_DATA_PATH, GOLD_DATA_PATH)
    print("--- Gold Layer Complete ---")

    print("--- Starting Model Training ---")
    
    # Load the final, prepared data from the Gold layer
    gold_data = pd.read_parquet(GOLD_DATA_PATH)
    
    # Separate features and target
    # The last columns of the gold_data are the features. The first two are the loan_application_id and label
    X = gold_data.drop(['loan_application_id', 'label'], axis=1)
    y = gold_data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile your model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model training complete. Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    run_full_pipeline()
