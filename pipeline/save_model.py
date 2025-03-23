import joblib
import argparse

def save_model(input_model_path, output_model_path):
    # Load the trained model
    model = joblib.load(input_model_path)

    # Save the model with a new name
    joblib.dump(model, output_model_path)
    print(f"✅ Model saved successfully as {output_model_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save trained model with a new name")
    parser.add_argument("--input_model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save the final model")

    args = parser.parse_args()
    
    save_model(args.input_model, args.output_model)
