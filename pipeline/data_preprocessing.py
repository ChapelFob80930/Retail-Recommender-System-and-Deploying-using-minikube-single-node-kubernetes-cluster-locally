import pandas as pd

def preprocess_data():
    df = pd.read_excel("/app/data/online_retail_II.xlsx")
    df['Total Price'] = df['Quantity'] * df['Price']
    preprocessed_dataset = df[df['Total Price'] > 0].copy()
    preprocessed_dataset = preprocessed_dataset.dropna(subset=['Customer ID', 'StockCode', 'Total Price'])
    preprocessed_dataset['Normalized Price'] = (
        5 * (preprocessed_dataset['Total Price'] - preprocessed_dataset['Total Price'].min()) /
        (preprocessed_dataset['Total Price'].max() - preprocessed_dataset['Total Price'].min())
    )
    preprocessed_dataset.to_csv("/app/data/preprocessed_data.csv", index=False)
    
