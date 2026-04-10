# Heart Disease Predictor

A Machine Learning laboratory project that predicts the presence of heart disease using a Logistic Regression model. The project features a modern, interactive Streamlit dashboard for real-time analysis.

## Features
- **Exploratory Data Analysis**: Pre-processing and visualization of the Statlog Heart dataset.
- **Machine Learning**: Implementation of Logistic Regression, Random Forest, SVM, and KNN (Logistic Regression used for final app).
- **Interactive UI**: Clean Streamlit dashboard with custom CSS and real-time probability visualization.
- **Scalable**: Decoupled training and inference scripts.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aarjav686/Heart_Disease_Predictor_ML_Lab.git
   cd Heart_Disease_Predictor_ML_Lab
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the trainer (optional, if you want to rebuild the model):
   ```bash
   python train_and_save.py
   ```

4. Launch the application:
   ```bash
   streamlit run app.py
   ```

## Dataset
This project uses the Statlog (Heart) dataset.
- Target: 0 = Absence, 1 = Presence of heart disease.
- Features: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, etc.

## Project Structure
- `ML_Lab_Project.ipynb`: Background analysis and model training exploration.
- `train_and_save.py`: Script to train the final model and export artifacts.
- `app.py`: Streamlit application file.
- `requirements.txt`: Python package requirements.
- `statlog+heart/`: Raw dataset directory.
