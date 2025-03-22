import joblib

def save_model():
    model = joblib.load("/app/models/SVD.pkl")
    joblib.dump(model, "/app/models/final_recommendation_model.pkl")
    print("Model saved successfully as final_recommendation_model.pkl!")
    
    