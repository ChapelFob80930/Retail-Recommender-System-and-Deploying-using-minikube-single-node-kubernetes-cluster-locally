import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import GridSearchCV
import joblib

def tune_hyperparameters():

    preprocessed_dataset = pd.read_csv("data/preprocessed_data.csv")
    

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(preprocessed_dataset[['Customer ID', 'StockCode', 'Normalized Price']], reader)
    

    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.02, 0.1]
    }


    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)


    best_params = grid_search.best_params['rmse']
    

    with open("data/best_params.txt", "w") as f:
        f.write(str(best_params))
    
    print(f"✅ Best hyperparameters saved: {best_params}")


    best_model = SVD(
        n_factors=best_params['n_factors'],
        n_epochs=best_params['n_epochs'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all']
    )


    trainset = data.build_full_trainset()
    best_model.fit(trainset)


    joblib.dump(best_model, "models/SVD_model.pkl")
    print("✅ Model retrained and saved as 'SVD_model.pkl'")
