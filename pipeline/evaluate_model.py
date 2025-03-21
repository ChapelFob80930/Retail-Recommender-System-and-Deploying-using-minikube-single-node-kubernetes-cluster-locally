import pandas as pd
import joblib
from surprise import Reader, Dataset, accuracy

def evaluate_model():
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
