# Recitation 4 || MLFlow Guide

MLFlow- an open-source MLOps platform designed to help manage end-to-end machine learning models and their lifecycle

### -  Create a new Python environment: -  
conda create -n recitation4 python=3.9 numpy pandas matplotlib \
conda activate recitation4 

### -  Install scikit-learn and MLFlow: - 
pip install scikit-learn \
pip install mlflow 

### -  Write your MLFlow workflow and start the MLFlow server: 
mlflow server --host localhost --port $PORT \
python main.py 