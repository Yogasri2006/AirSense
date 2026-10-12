# AirSense
AI-Based Air Quality Analysis & PM2.5 Prediction Web Application.

## Setup (Windows PowerShell)
```powershell
cd path\to\AirSense
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Put `PRSA_Data_Aotizhongxin_20130301-20170228.csv` in `dataset`.

Train:
```powershell
python train_model.py
```
Run:
```powershell
python app.py
```
Open http://127.0.0.1:5000/

The model uses a time-aware 80/20 split and a single saved scikit-learn pipeline so training and prediction use the same preprocessing.
