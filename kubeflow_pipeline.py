import time
import kfp
from kfp.dsl import component, pipeline
import pandas as pd



@component(base_image="chapelfob80930/retail-recommendation-pipeline:latest")
def data_preprocessing():
    import sys
    sys.path.append('/app')
    import pipeline.data_preprocessing
    pipeline.data_preprocessing.preprocess_data()

@component(base_image="chapelfob80930/retail-recommendation-pipeline:latest")
def model_training():
    import sys
    sys.path.append('/app')
    import pipeline.model_training
    pipeline.model_training.train_model()

@component(base_image="chapelfob80930/retail-recommendation-pipeline:latest")
def hyperparameter_tuning():
    import sys
    sys.path.append('/app')
    import pipeline.hyperparameter_tuning
    pipeline.hyperparameter_tuning.tune_hyperparameters()    

@component(base_image="chapelfob80930/retail-recommendation-pipeline:latest")
def evaluate_model():
    import sys
    sys.path.append('/app')
    import pipeline.evaluate_model
    pipeline.evaluate_model.evaluate_model()

@component(base_image="chapelfob80930/retail-recommendation-pipeline:latest")
def save_model():
    import sys
    sys.path.append('/app')
    import pipeline.save_model
    pipeline.save_model.save_model()

@pipeline(name="Retail Recommendation Pipeline", description="Pipeline to build and deploy a recommendation model")
def recommendation_pipeline():
    data_preprocessing_task = data_preprocessing()
    model_training_task = model_training().after(data_preprocessing_task)
    hyperparameter_tuning_task = hyperparameter_tuning().after(data_preprocessing_task)
    evaluation_task = evaluate_model().after(model_training_task, hyperparameter_tuning_task)
    save_model_task = save_model().after(evaluation_task)

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(recommendation_pipeline, "retail_recommendation_pipeline.yaml")
    
    print("Pipeline executed successfully!")
    # while True:
    #     time.sleep(60)
    
