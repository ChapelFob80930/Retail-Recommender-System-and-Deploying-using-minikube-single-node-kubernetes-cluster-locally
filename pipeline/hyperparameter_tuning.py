import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import GridSearchCV
import joblib
import argparse

def tune_hyperparameters(input_path, output_model, output_params):

    # Load dataset
    preprocessed_dataset = pd.read_csv(input_path)
    
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(preprocessed_dataset[['Customer ID', 'StockCode', 'Normalized Price']], reader)

    # Define hyperparameter grid
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.02, 0.1]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)

    best_params = grid_search.best_params['rmse']

    # Save best hyperparameters
    with open(output_params, "w") as f:
        f.write(str(best_params))
    
    print(f"✅ Best hyperparameters saved to {output_params}: {best_params}")

    # Train the best model
    best_model = SVD(
        n_factors=best_params['n_factors'],
        n_epochs=best_params['n_epochs'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all']
    )

    trainset = data.build_full_trainset()
    best_model.fit(trainset)

    # Save the trained model
    joblib.dump(best_model, output_model)
    print(f"✅ Model retrained and saved as {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SVD model.")
    parser.add_argument("--input", type=str, required=True, help="Path to the preprocessed dataset CSV")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--output_params", type=str, required=True, help="Path to save best hyperparameters")

    args = parser.parse_args()
    
    tune_hyperparameters(args.input, args.output_model, args.output_params)
