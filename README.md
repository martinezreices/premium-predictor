⚕️ Insurance Premium Predictor
This is an end-to-end machine learning project that predicts medical insurance premiums based on an individual's attributes. The project includes data preprocessing, model training with hyperparameter tuning, and a fully interactive web application built with Streamlit for real-time predictions.

🚀 Live Demo & Screenshot

[**➡️ View the Live Application Here**](https://health-insurance-premium-predictor-martinezreices.streamlit.app/)

📖 Project Overview
The goal of this project is to build and deploy a reliable model that can estimate a person's annual and monthly medical insurance costs. This demonstrates a complete machine learning workflow, from initial data exploration to a final, user-facing product.

The key stages of the project were:

Data Preprocessing: The initial dataset was loaded and cleaned. A ColumnTransformer was used to create a preprocessing pipeline that handles both numerical features (using StandardScaler) and categorical features (using OneHotEncoder).

Model Training: An XGBoost Regressor model was chosen for its high performance. The model was trained within a full scikit-learn pipeline to ensure robustness and prevent data leakage.

Hyperparameter Tuning: GridSearchCV was employed to automatically test various combinations of model parameters (n_estimators, max_depth, learning_rate) and find the optimal set, maximizing the model's predictive accuracy.

Model Persistence: The final, best-performing model pipeline was saved to a premium_predictor_model.pkl file using pickle, allowing it to be easily loaded for inference without retraining.

Interactive Web App: A user-friendly web application was built using Streamlit. The app features an intuitive sidebar where users can input their details.

Deployment: The application was configured for cloud deployment, with specific library and Python versions defined in requirements.txt and runtime.txt to ensure a stable and reproducible environment on Streamlit Community Cloud.

✨ Key Features
Real-Time Predictions: The app provides instant annual and monthly premium estimates when the user clicks "Submit".

Automatic BMI Calculation: Users enter their height and weight, and the app automatically calculates their BMI, providing a better user experience.

State-to-Region Mapping: Users select their state, and the app intelligently maps it to the correct geographical region (southwest, southeast, northwest, northeast) that the model uses.

Intuitive Interface: A clean, sidebar-based layout makes it easy for anyone to input their data and get a prediction.

🛠️ Technologies Used
Language: Python

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Web Framework: Streamlit

Model Persistence: Pickle

📂 Project Structure
The project is organized into a clean, modular structure for maintainability:

├── data/
│   └── insurance.csv         # Raw dataset
├── src/
│   ├── train.py              # Script to preprocess data, train the model, and save the .pkl file
│   └── premium_predictor_model.pkl # The saved, trained model pipeline
├── .gitignore                # Files to be ignored by Git
├── app.py                    # The main Streamlit application script
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Required Python libraries for deployment
└── runtime.txt               # Specified Python version for deployment

⚙️ How to Run Locally
To run this application on your own machine, please follow these steps:

Clone the Repository:

git clone [https://github.com/martinezreices/premium-predictor](https://github.com/martinezreices/premium-predictor)
cd premium-predictor

Create and Activate a Virtual Environment:

# Create the environment
python -m venv venv

# Activate it (on Mac/Linux)
source venv/bin/activate

# Or on Windows
.\venv\Scripts\activate

Install Dependencies: Ensure your requirements.txt and runtime.txt files are present, then run:

pip install -r requirements.txt

Generate the Model File: If the premium_predictor_model.pkl file is not present, run the training script to create it:

python src/train.py

Run the Streamlit App:

streamlit run app.py
