# Sonar Rock vs Mine Prediction (Machine Learning Project)

This is a machine learning project that predicts whether an object is a **Rock (R)** or **Mine (M)** based on sonar signal data.  
It uses the **Logistic Regression** algorithm and provides a simple **Tkinter GUI** to allow users to input their data and get predictions.

---

 Project Structure
SONAR-PROJECT/
├── sonar.py # ML training code
├── rock_mine_model.sav # Trained model (saved using joblib)
├── rockvsmine.py # GUI for predicting user input
├── sonar data.csv # Dataset (UCI Sonar)
└── README.md # This file


- Model Used: Logistic Regression  
- Library: scikit-learn  
- Dataset: Sonar Data from UCI ML Repository  
- Preprocessing: No standardization or scaling used (values were already normalized between 0 and 1)  
- Accuracy:
  - Training Accuracy: ~ 0.8342245989304813
  - Testing Accuracy: ~ 0.7619047619047619

> Note: The dataset contains 60 numeric sonar readings per row with label `R` (Rock) or `M` (Mine).

---

 GUI Usage

- Built using tkinter
- Enter 60 comma-separated values in the text box
- Click "Predict" to get whether the object is a Rock or Mine

 How to Run :- 
  - launch rockvsmine.py

