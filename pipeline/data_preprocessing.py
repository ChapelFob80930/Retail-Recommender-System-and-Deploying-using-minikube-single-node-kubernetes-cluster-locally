import pandas as pd
import argparse

def preprocess_data(input_path, output_path):
    # Load dataset dynamically from the given input path
    df = pd.read_excel(input_path)

    # Data preprocessing
    df['Total Price'] = df['Quantity'] * df['Price']
    preprocessed_dataset = df[df['Total Price'] > 0].copy()
    preprocessed_dataset = preprocessed_dataset.dropna(subset=['Customer ID', 'StockCode', 'Total Price'])

    # Normalize price column
    preprocessed_dataset['Normalized Price'] = (
        5 * (preprocessed_dataset['Total Price'] - preprocessed_dataset['Total Price'].min()) /
        (preprocessed_dataset['Total Price'].max() - preprocessed_dataset['Total Price'].min())
    )

    # Save preprocessed data dynamically to the given output path
    preprocessed_dataset.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

# Allow running the script with custom file paths
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for recommendation system.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Excel dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to save preprocessed CSV data.")
    
    args = parser.parse_args()
    preprocess_data(args.input, args.output)
