# ⚽ Premier League Match Outcome Predictor

This project is a Machine Learning application that predicts the outcome of Premier League matches using historical data.

The model predicts one of three outcomes:
- Home Win (H)
- Draw (D)
- Away Win (A)

---

## 📊 Features

- Uses multiple seasons of EPL data
- Feature engineering based on:
  - Recent team form
  - Goals scored and conceded
- Logistic Regression model (multiclass classification)
- Interactive web app built with Streamlit

---

## 📁 Project Structure

Prem_Match_Predictor/

app.py                  # Streamlit app (UI)  
train_model.py          # Model training script  
requirements.txt        # Dependencies  

model.pkl               # Trained model (generated)  
feature_columns.pkl     # Feature metadata (generated)  
teams.pkl               # Team list (generated)  
processed_matches.csv   # Processed dataset (generated)  

data/                   # Raw CSV data  
  EPL_2021.csv  
  EPL_2022.csv  
  EPL_2023.csv  
  ...  

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/RuiC2005/prem-match-predictor.git
cd prem-match-predictor

---

### 2. Install requirements 
pip install -r requirements.txt


### 3. Add your data

Place your EPL CSV files inside the data/ folder.

Each file must contain the following columns:

Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR

### 4. Train the model
python train_model.py

This will generate:

model.pkl
feature_columns.pkl
teams.pkl

### 5. Run the app
streamlit run app.py 
