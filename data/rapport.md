# Telecom Churn Analysis Report

## 1. Summary scikit-learn

The goal of this study was to identify customers at risk of churn for a telecom company and propose retention actions.

We used a dataset containing demographic, service, and financial information for each customer. Three classification models were tested: **Logistic Regression, Random Forest, and Gradient Boosting**.

The **Logistic Regression model**, optimized with GridSearchCV, was selected as the best model based on the **recall**, due to its ability to handle categorical variables and class imbalance.

---

## 2. Methodology

### 2.1 Data Exploration and Preparation
- Loaded and inspected the dataset (`03_DONNEES.csv`).  
- Missing values were handled: **median** for numeric variables, **mode** for categorical variables.  
- Selected features: `gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, InternetCharges, MonthlyCharges, TotalCharges`.  
- Categorical variables were encoded using **OneHotEncoder**, numeric variables were scaled using **StandardScaler**.  
- Data split into **train/test** sets (70/30).

### 2.2 Modeling
- Three models evaluated: **Logistic Regression, Random Forest, Gradient Boosting**.  
- Metrics computed: **accuracy, precision, recall, F1-score**.  
- Best model selected based on **recall**: Logistic Regression.  
- Extracted **top 10 most important features** for business interpretation.  
- Performed **stratified 5-fold cross-validation** to assess robustness.  
- Optimized hyperparameters using **GridSearchCV**.

---

## 3. Results

### 3.1 Model Performance
| Model                | Accuracy | Precision | Recall  | F1-score |
|----------------------|---------|----------|--------|----------|
| Logistic Regression  | 0.6313  | 0.2309   | 0.6890 | 0.3459   |
| Random Forest        | 0.8566  | 0.4091   | 0.0301 | 0.0561   |
| Gradient Boosting    | 0.8547  | 0.2500   | 0.0134 | 0.0254   |

> **Logistic Regression** achieved the highest **recall**, which is suitable for imbalanced class problems.

### 3.2 Top Important Features
The top 10 features influencing churn prediction are:  
- `Contract` (subscription type)  
- `tenure` (customer tenure)  
- `MonthlyCharges`  
- `InternetService`  
- `TechSupport`  
- `OnlineSecurity`  
- `StreamingMovies`  
- `DeviceProtection`  
- `Partner`  
- `Dependents`  

These features can help target high-risk customers and design retention strategies.

### 3.3 High-Risk Customer Identification
- Selected **20 customers with churn probability > 80%**.  
- Typical profile: **Month-to-Month contract**, tenure < 12 months, high monthly charges, limited support services.

---

## 4. Business Recommendations

1. **Targeted Promotional Offers**  
   - Discounts or bundled services for high-risk customers.  

2. **Engagement / Loyalty Programs**  
   - Bonuses, free months, or service upgrades to extend subscriptions.  

3. **Personalized Follow-Up Calls**  
   - Proactive calls to detect issues and offer solutions to retain customers.  

### 4.1 Estimated ROI
- Average cost per action: €20 per customer.  
- Expected gain: average **MonthlyCharges × 3 months of retention**.  
- Total estimated ROI for 20 high-risk customers: **≈ €1,500**, indicating the retention actions are cost-effective.

---

## 5. Generated Files

- `predictions_test.csv` → Churn predictions on the test set.  
- `model_metrics.json` → Accuracy, F1, precision, and recall of the best model.  
- `feature_importance.csv` → Feature importance for Random Forest.

## 6. Pyspark Summary (03_bonus.ipynb)

All previous steps still apply, but this time, **Random Forest** was the best-performing model (but all model is almost equal).

### 5b - Spark Preparation

| Model                | Accuracy | Precision | Recall  | F1-score | Time (s) |
|----------------------|---------|----------|--------|----------|----------|
| Logistic Regression  | 0.8575  | 0.7837   | 0.8575 | 0.7931   | 3.6994   |
| Random Forest        | 0.8580  | 0.7362   | 0.8580 | 0.7924   | 2.2006   |
| Gradient Boosting    | 0.8535  | 0.8029   | 0.8535 | 0.8092   | 19.4381  |

**Conclusion from comparing scikit-learn and PySpark (03_bonus.ipynb):**  
We can see that, although there are slight improvements in metrics, the runtime for Gradient Boosting is clearly very high.
