import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
import joblib
import argparse

def train_model(input_path, output_model):
    # Load preprocessed dataset
    preprocessed_dataset = pd.read_csv(input_path)
    
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(preprocessed_dataset[['Customer ID', 'StockCode', 'Normalized Price']], reader)
    
    # Split dataset into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train the model
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    
    # Save trained model
    joblib.dump(model, output_model)
    print(f"✅ Model trained and saved at {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVD Model")
    parser.add_argument("--input", type=str, required=True, help="Path to the preprocessed dataset CSV")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()
    
    train_model(args.input, args.output_model)