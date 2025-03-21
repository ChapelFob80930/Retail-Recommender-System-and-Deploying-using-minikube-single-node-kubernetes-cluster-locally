import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
import joblib

def train_model():
    preprocessed_dataset = pd.read_csv("/Data/preprocessed_data.csv")
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(preprocessed_dataset[['Customer ID', 'StockCode', 'Normalized Price']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    joblib.dump(model, "/models/SVD_model.pkl")
