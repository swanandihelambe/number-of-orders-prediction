# Number of Orders Prediction using Machine Learning

This project focuses on predicting the **daily number of customer orders per product category** using machine learning techniques on the Brazilian Olist e‑commerce dataset.

---

## Project Walkthrough

1. Load and merge raw Olist orders, order items, and products tables  
2. Clean and preprocess data; aggregate to daily order counts  
3. Engineer time, marketing, and demand‑based features (lags & moving averages)  
4. Encode categorical variables and normalize numerical features  
5. Split data into train and test sets  
6. Train Linear Regression, Random Forest, and XGBoost regression models  
7. Evaluate models using RMSE, MAE, and R² metrics  
8. Visualize demand patterns and actual vs predicted order counts  
9. Save the final XGBoost model for batch or real‑time inference (optional)

---

## Dataset

- **Source:** Brazilian E‑Commerce Public Dataset by Olist (Kaggle).
- **Tables used:**  
  - `olist_orders_dataset`  
  - `olist_order_items_dataset`  
  - `olist_products_dataset`  
- **Target:**  
  - `order_count` = number of unique orders per product category per day  
- **Key features:**  
  - Date: `order_date`  
  - Category: `product_category_name`  
  - Price and freight: `avg_price`, `avg_freight`  
  - Time features: `day_of_week`, `month`, `year`, `is_weekend`  
  - Marketing: `discount`, `campaign`  
  - Demand history: `lag_1`, `ma_7` (7‑day moving average)

---

## Methods

### 1. Data Cleaning & Preprocessing

- Merged orders, order items, and product tables on `order_id` and `product_id`.  
- Converted timestamps to `order_date` and aggregated to daily order counts per category.  
- Dropped rows with missing lag values and kept relevant numeric columns.  
- One‑hot encoded `product_category_name` for modeling and normalized selected numeric features using MinMax scaling.

### 2. Feature Engineering

- Time‑based features to capture seasonality: `day_of_week`, `month`, `year`, `is_weekend`.  
- Synthetic promotion features: `discount` and `campaign` flags.  
- Lag and moving‑average features (`lag_1`, `ma_7`) to incorporate recent demand history for each category.

### 3. Modeling & Evaluation

- Train/test split: 80% training, 20% testing.  
- Models trained:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor  
- Metrics:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² score  
- XGBoost achieved the best performance and was selected as the final model for order‑count prediction.

---

## Results & Visualizations

- EDA plots showing:
  - Distribution of daily order counts  
  - Average orders by day of week  
  - Top product categories by total orders  
- Actual vs Predicted plot for XGBoost on the test set, illustrating how well the model follows real demand over time and where it misses extreme spikes.

---

## Deployment

- The final XGBoost model is saved using `joblib` as `xgb_number_of_orders_model.pkl`.  
- It can be loaded in a separate script or API to support batch or real‑time demand forecasting for logistics, inventory, and marketing planning.[file:1]
