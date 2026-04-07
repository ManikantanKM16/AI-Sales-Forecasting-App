# 📊 E-Commerce Sales Forecasting & Scarcity Analysis
> **Predictive Modeling | Statistical Inference | Interactive Deployment**

## 🎯 Project Overview
This project investigates the impact of **scarcity marketing** (Limited vs. Vast stock quantity) on e-commerce sales performance. It moves beyond simple data cleaning to provide a full-stack Data Science solution: from statistical validation of consumer behavior to a deployed machine learning model for sales forecasting.

---

## 🚀 Key Features
* **Statistical Rigor:** Conducted **Welch’s T-Test** to validate the "Scarcity Effect," proving that limited stock labels drive a statistically significant lift in conversion ($p < 0.05$).
* **Machine Learning:** Developed a **Random Forest Regressor** to forecast sales volume based on price, ratings, and inventory status.
* **Feature Engineering:** Implemented One-Hot Encoding for categorical variables and Log Transformation for target normalization to handle skewed sales data.
* **Interactive Dashboard:** Built a **Streamlit Web App** that allows stakeholders to perform "What-If" analysis on pricing and stock strategies.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Scipy, Matplotlib, Seaborn
* **Deployment:** Streamlit, Joblib
* **Version Control:** Git & GitHub

---

## 📈 Model Performance & Insights
* **Top Predictors:** Feature importance analysis revealed that **Price** and **Stock Status** are the primary drivers of sales volume.
* **Accuracy:** The model achieved an **R² score of [Insert Your R2 Score here, e.g., 0.84]**, indicating high predictive reliability.
* **Business Impact:** Identified a "Price Ceiling" where scarcity no longer compensates for high costs, allowing for optimized discount strategies.

---

## 🖥️ How to Run the App
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AI-Sales-Forecasting-App.git](https://github.com/YOUR_USERNAME/AI-Sales-Forecasting-App.git)
