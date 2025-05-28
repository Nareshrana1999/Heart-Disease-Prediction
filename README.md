# ❤️ Heart Disease Prediction

A comprehensive machine learning web application that predicts the risk of heart disease based on patient health metrics. The system employs multiple ML models to provide accurate and reliable predictions, helping in early detection and risk assessment of heart disease.

## ✨ Key Features

### 🧠 Advanced Machine Learning
- **Ensemble Learning**: Combines predictions from 8 different ML models for robust results
- **Model Diversity**: Utilizes various algorithms including Random Forest, XGBoost, and Neural Networks
- **Continuous Learning**: Easy to retrain models with new data

### 💻 User Experience
- **Intuitive Interface**: Clean, modern web interface for seamless interaction
- **Real-time Feedback**: Instant risk assessment with detailed probability scores
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Interactive Visuals**: Clear graphical representation of risk factors

### 🔍 Detailed Analysis
- **Comprehensive Risk Assessment**: Evaluates multiple health parameters
- **Feature Importance**: Highlights key factors affecting the prediction
- **Confidence Scores**: Shows model confidence for each prediction

### ⚙️ Technical Excellence
- **RESTful API**: Easy integration with other systems
- **Modular Codebase**: Well-structured and documented code
- **Pre-trained Models**: Ready-to-use models included
- **Scalable Architecture**: Can handle multiple concurrent users

## 📁 Project Structure

```
Heart Disease Prediction/
│
├── app.py                 # Main Flask application
├── train.py              # Model training and evaluation
├── test_load.py          # Data loading test script
├── data_preprocessing.py # Data cleaning and preprocessing
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
├── data/                # Data directory
│   └── heart_disease_cleveland.csv  # Dataset
├── models/              # Trained models and scaler
│   ├── decision_tree_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── knn_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── svm_model.pkl
│   └── xgboost_model.pkl
└── templates/
    └── index.html        # Web interface
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/Nareshrana1999/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Set Up Virtual Environment (Recommended)
#### Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset
- Place your dataset in the `data/` directory as `heart_disease.csv`
- Alternatively, use the provided sample dataset

### 5. Train Models (Optional)
Pre-trained models are included, but you can retrain them:
```bash
python train.py
```

### 6. Launch the Application
```bash
python app.py
```

### 7. Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 8. Using the Application
1. Fill in the patient's health metrics
2. Click 'Predict' to get the risk assessment
3. View detailed results and risk factors

## 🤖 Machine Learning Models

The application utilizes the following algorithms:

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble learning method that operates by constructing multiple decision trees |
| XGBoost | Optimized distributed gradient boosting library |
| Logistic Regression | Statistical model for binary classification |
| Gradient Boosting | Builds an additive model in a forward stage-wise fashion |
| Support Vector Machine (SVM) | Finds the optimal hyperplane for classification |
| K-Nearest Neighbors (KNN) | Classifies based on the k-nearest data points |
| Naive Bayes | Probabilistic classifier based on Bayes' theorem |
| Decision Tree | Tree-like model of decisions and their possible consequences |

## 📊 Dataset

The model is trained on the Cleveland Heart Disease dataset from the UCI Machine Learning Repository, containing 14 attributes including:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise induced angina
- ST depression
- Slope of the peak exercise ST segment
- Number of major vessels
- Thalassemia

## 🛠️ Customization

### Training Parameters
You can modify the training parameters in `train.py` to adjust model behavior:
- Number of estimators
- Maximum depth
- Learning rate
- Cross-validation settings

### Web Interface
Customize the web interface by editing files in the `templates/` directory:
- `index.html`: Main interface
- CSS styling can be modified directly in the HTML file


## 🤖 Machine Learning Models

The application uses the following models:

- Random Forest (Best Performing)
- XGBoost
- Logistic Regression
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree

## 📊 Dataset

Trained on the Cleveland Heart Disease dataset from UCI Machine Learning Repository.

## 🤖 Machine Learning Models

The system utilizes an ensemble of the following machine learning models:

1. Random Forest Classifier
2. XGBoost Classifier
3. Gradient Boosting Classifier
4. Logistic Regression
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Naive Bayes
8. Decision Tree

### Ensemble Approach
The final prediction combines the probabilities from all models using a weighted average, where weights are based on each model's cross-validation accuracy. This approach helps to reduce variance, improve generalization, and provide more robust predictions.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Permissions
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

### Limitations
- ❌ Liability
- ❌ Warranty
- ❌ Trademark use

### Conditions
- ℹ️ Include original license and copyright notice
- ℹ️ State changes made to the original code

For more information, please refer to the [MIT License](https://choosealicense.com/licenses/mit/).

## 📧 Contact

Naresh Rana - Nareshrana1999@outlook.com


## Disclaimer

This tool provides risk assessment only and is not a substitute for professional medical advice. Always consult with a healthcare provider for medical diagnosis and treatment.