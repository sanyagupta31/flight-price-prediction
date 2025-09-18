

**Project Title:** Flight Price Prediction (Indian Domestic Flights)

**Overview:**
This project predicts the ticket prices of Indian domestic flights (March–June 2019).
The dataset includes multiple features such as airline, source, destination, journey date, duration, and number of stops.
The goal is to apply data preprocessing, feature engineering, and machine learning models to estimate flight ticket prices accurately.

---

**Project Structure:**

* `data/` → contains the dataset (flights.csv)
* `models/` → stores the trained ML model (`flight_price_model.pkl`) and evaluation metrics (`metrics.txt`)
* `notebooks/` → includes `train.py` for model training and evaluation
* `app/` → contains `app.py`, the Streamlit application for predictions
* `requirements.txt` → lists required Python libraries
* `README.md` → project documentation

---

**Setup Instructions:**

1. Clone the repository:
   `git clone https://github.com/sanyagupta31/flight-price-prediction.git`
   `cd flight-price-prediction`

2. Create a virtual environment and install dependencies:

   * Windows:
     `python -m venv venv`
     `venv\Scripts\activate`
   * Mac/Linux:
     `python -m venv venv`
     `source venv/bin/activate`
   * Install dependencies:
     `pip install -r requirements.txt`

3. Train the model:
   `python notebooks/train.py`
   This will generate:

   * Trained model file: `models/flight_price_model.pkl`
   * Metrics file: `models/metrics.txt`

4. Run the Streamlit application:
   `streamlit run app/app.py`

5. Open the application in your browser at:
   `http://localhost:8501`

---

**Model Performance (Example):**
The following metrics are stored in `models/metrics.txt`:

* RMSE: XXXX
* MAE: XXXX
* R²: XXXX

---

**Features & Methodology:**

* Feature Engineering:

  * Extracted journey day and month from date
  * Converted duration to minutes
  * Encoded categorical variables (airline, source, destination)

* Model:

  * Random Forest Regressor with hyperparameter tuning (RandomizedSearchCV)

* Evaluation:

  * Metrics used: RMSE, MAE, R²

---

**Competition Information:**

* Hosted by: Analytical Arena – Data Science Club
* Dataset: Indian Domestic Flights (March–June 2019)
* Objective: Predict flight ticket prices

---
Here is my streamlit link: https://sanyagupta31-flight-price-prediction-appapp-wv8bhc.streamlit.app/
---
**License:**
This project is for educational purposes as part of the Analytical Arena Data Science Challenge.

---


