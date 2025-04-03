# Retail Recommender System using SVD 🛒

This project builds a personalized recommendation system using Singular Value Decomposition (SVD) on normalized price data (0-5 scale). The model predicts customer purchase behavior with high accuracy (RMSE: 0.0199, MAE: 0.0051). The code for this project was developed using Visual Studio Code (VS Code).

## Features ✨

- **Data Preprocessing and Normalization** 📊
  - Handles data cleaning and normalization processes to prepare the dataset for modeling.

- **Model Training and Evaluation** 🧠
  - Implements various collaborative filtering algorithms including SVD, NMF, and SVD++.
  - Evaluates the models using metrics such as RMSE and MAE.

- **Hyperparameter Tuning** 🔧
  - Uses GridSearchCV to find the best hyperparameters for the models.

- **Scalability** 📈
  - Supports predictions for new users and items, making the system scalable and adaptable to changes in the dataset.

## Final Model 🚀

The final model is an SVD model with normalized price, which is ready for deployment.

## Getting Started 🛠️

### Prerequisites

- Python 3.x 
- Jupyter Notebook 
- Minikube 
- Docker 🐋

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ChapelFob80930/Retail-Recommender-System-and-Deploying-using-minikube-single-node-kubernetes-cluster-locally.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Retail-Recommender-System-and-Deploying-using-minikube-single-node-kubernetes-cluster-locally
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Open Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Navigate to the notebook file and run the cells to train and evaluate the recommendation system.

### Deployment 🚀

To deploy the recommendation system using Minikube on a single-node Kubernetes cluster, follow these steps:

1. Start Minikube:
    ```bash
    minikube start
    ```

2. Build the Docker image:
    ```bash
    docker build -t chapelfob80930/retail-recommendation-pipeline:latest .
    ```

3. Push the Docker image to Docker Hub:
    ```bash
    docker push chapelfob80930/retail-recommendation-pipeline:latest
    ```

4. Run the Docker container:
    ```bash
    docker run chapelfob80930/retail-recommendation-pipeline:latest
    ```

5. Create the pipeline:
    ```bash
    kfp pipeline create -p retail_recommendation retail_recommendation_pipeline.yaml
    kfp pipeline create-version ./retail_recommendation_pipeline.yaml --pipeline-id 25130bec-5243-4b5f-92ba-c6ab9b143661 --pipeline-version "retail_recommendation_v2"
    ```

6. Expose Kubeflow to a port:
    ```bash
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```

## Contributing 🤝

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License 📄

This project is licensed under the MIT License. See the `LICENSE` file for more details.
