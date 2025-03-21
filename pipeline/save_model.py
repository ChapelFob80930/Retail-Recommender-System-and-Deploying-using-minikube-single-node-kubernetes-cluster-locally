import joblib

def save_model():
    model = joblib.load("models/SVD_model.pkl")
    joblib.dump(model, "models/final_recommendation_model.pkl")
    print("Model saved successfully as final_recommendation_model.pkl!")
    
    