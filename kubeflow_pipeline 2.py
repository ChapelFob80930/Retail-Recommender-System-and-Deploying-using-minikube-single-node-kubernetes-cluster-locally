import kfp
from kfp.dsl import component, pipeline
import pandas as pd
import joblib
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split

@component
def data_preprocessing(base_image: str = "retail-recommendation-pipeline:latest"):
    df = pd.read_excel("/data/online_retail_II.xlsx")
    df['Total Price'] = df['Quantity'] * df['Price']
    preprocessed_dataset = df[df['Total Price'] > 0].copy()
    preprocessed_dataset = preprocessed_dataset.dropna(subset=['Customer ID', 'StockCode', 'Total Price'])
    preprocessed_dataset['Normalized Price'] = (
        5 * (preprocessed_dataset['Total Price'] - preprocessed_dataset['Total Price'].min()) /
        (preprocessed_dataset['Total Price'].max() - preprocessed_dataset['Total Price'].min())
    )
    preprocessed_dataset.to_csv("/data/preprocessed_data.csv", index=False)

@component
def model_training(base_image: str = "retail-recommendation-pipeline:latest"):
    preprocessed_dataset = pd.read_csv("/Data/preprocessed_data.csv")
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(preprocessed_dataset[['Customer ID', 'StockCode', 'Normalized Price']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    joblib.dump(model, "/models/SVD_model.pkl")

@component
def hyperparameter_tuning(base_image: str = "retail-recommendation-pipeline:latest"):
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

@component
def evaluate_model(base_image: str = "retail-recommendation-pipeline:latest"):
    preprocessed_dataset = pd.read_csv("/data/preprocessed_data.csv")
    model = joblib.load("/models/SVD_model.pkl")

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(preprocessed_dataset[['Customer ID', 'StockCode', 'Normalized Price']], reader)
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    with open("/data/metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")


@component
def save_model(base_image: str = "retail-recommendation-pipeline:latest"):
    model = joblib.load("models/SVD_model.pkl")
    joblib.dump(model, "models/final_recommendation_model.pkl")
    print("Model saved successfully as final_recommendation_model.pkl!")

@pipeline(name="Retail Recommendation Pipeline", description="Pipeline to build and deploy a recommendation model")
def recommendation_pipeline():
    data_preprocessing_task = data_preprocessing()
    model_training_task = model_training().after(data_preprocessing_task)
    hyperparameter_tuning_task = hyperparameter_tuning().after(data_preprocessing_task)
    evaluation_task = evaluate_model().after(model_training_task, hyperparameter_tuning_task)
    save_model_task = save_model().after(evaluation_task)

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(recommendation_pipeline, "retail_recommendation_pipeline_2.yaml")
