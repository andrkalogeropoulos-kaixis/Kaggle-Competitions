# Kaggle Machine Learning Competitions 🏆

This repository contains my solutions and code for various **Kaggle** competitions. These projects were developed as part of my learning journey and practical application of the following Kaggle Learn modules:
* *Intro & Intermediate Machine Learning*
* *Pandas & Data Visualization*
* *Feature Engineering* (Advanced techniques, Clustering)

---

## 🏠 Project 1: House Prices - Advanced Regression Techniques
* **Competition:** Housing Prices Competition for Kaggle Learn Users
* **Global Ranking:** **388 / 4,050 (Top 10%)** 🥇
* **Objective:** Predict the final sale price of residential homes using 79 explanatory variables.

### 🛠️ Technical Approach & Feature Engineering
* **Feature Construction:**
  * `TotalBath`: Combined full and half bathrooms into a single metric.
  * `AgeWhenSold`: Calculated the exact age of the property at the time of sale.
  * `Quality_x_Area`: Created an interaction feature multiplying overall quality by above-ground living area (which proved to be the model's most dominant feature).
* **Missing Value Handling:** Analyzed missing data and dropped columns with a missing value rate higher than 20%.
* **Data Pipelines:** Utilized `ColumnTransformer` to streamline preprocessing:
  * *Numerical Data:* Imputed missing entries using `SimpleImputer(strategy='median')`.
  * *Low Cardinality Categorical:* Applied `OneHotEncoder`.
  * *High Cardinality Categorical:* Applied `OrdinalEncoder`.
* **Modeling:** Trained an **XGBRegressor** (tuned with 214 estimators and a learning rate of 0.05).
* **Evaluation:** 5-Fold Cross-Validation yielding an **Average CV MAE of 14443.42790**.

---

## 🏎️ Project 2: F1 Pit Stop Prediction (Classification)
* **Competition:** Playground Series - Season 6, Episode 5 (Predicting F1 Pit Stops)
* **Global Ranking:** 4,265 / 5,022 *(Baseline Model - Learning Stage)*
* **Objective:** Binary classification to predict whether a driver will pit on the very next lap (`PitNextLap`).

### 🛠️ Technical Approach & Feature Engineering
* **Advanced Feature Engineering:**
  * Created complex mathematical interactions (e.g., `Tyre_vs_Lap`, `Degr_vs_Time`) utilizing cube roots to smooth heavily skewed distributions.
  * Outlier management using `np.clip` on highly volatile degradation and delta metrics.
* **Unsupervised Learning (K-Means Clustering):**
  * Scaled core racing metrics using `StandardScaler` and trained a **K-Means algorithm (5 clusters)** on features like `TyreLife` and `LapNumber`.
  * Generated distance-to-centroid features (`Centroid_0` through `Centroid_4`) and fed them directly into the supervised model as structural meta-features.
* **Modeling:** Deployed an **XGBClassifier** with `scale_pos_weight=10` to heavily penalize errors on the minority class, effectively resolving the severe class imbalance typical of pit stop events.
* **Evaluation:** 5-Fold Cross-Validation achieving a strong **ROC-AUC score of 0.94**.

---

## 🧰 Tech Stack & Libraries

* **Language:** Python 3
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Pipelines, Imputers, Encoders, KMeans, Scalers)
* **Gradient Boosting:** XGBoost (XGBRegressor, XGBClassifier)
* **Visualization:** Matplotlib, Seaborn

---

### 📂 Repository Structure
```text
├── House-Prices-Regression/
│   ├── house_prices_solution.ipynb  # Notebook for the Housing Competition
│   └── submission.csv
├── F1-Pit-Stops-Classification/
│   ├── f1_pit_stops_solution.ipynb  # Notebook for the F1 Playground
│   └── submission.csv
└── README.md
