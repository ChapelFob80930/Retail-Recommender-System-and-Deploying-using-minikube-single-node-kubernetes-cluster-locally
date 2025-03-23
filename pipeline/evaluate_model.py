import pandas as pd
import joblib
import argparse
from surprise import Reader, Dataset, accuracy

def evaluate_model(input_path, model_path, output_path):
    # Load dataset dynamically from the given input path
    preprocessed_dataset = pd.read_csv(input_path)

    # Load the trained model dynamically
    model = joblib.load(model_path)

    # Prepare data for evaluation
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(preprocessed_dataset[['Customer ID', 'StockCode', 'Normalized Price']], reader)
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    predictions = model.test(testset)

    # Calculate performance metrics
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Save metrics dynamically to the specified output path
    with open(output_path, "w") as f:
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
    
    print(f"✅ Metrics saved to {output_path}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

# Allow running the script with dynamic file paths
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained recommendation model.")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed CSV dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save evaluation metrics.")

    args = parser.parse_args()
    evaluate_model(args.input, args.model, args.output)
