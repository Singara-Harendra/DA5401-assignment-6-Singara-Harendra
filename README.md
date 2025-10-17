# Missing Data Imputation via Regression for Credit Card Default Prediction

This project explores and compares different strategies for handling missing data in the context of a credit card default risk assessment model. The primary goal is to determine the most effective imputation technique by evaluating its impact on the performance of a downstream classification model.

🧐 **Core Question:** How does the strategy for handling missing data (from simple median imputation to complex regression-based methods) affect the performance of a credit card default prediction model?

---

## 📊 Project Workflow

The project follows a structured experimental workflow:

1.  **Data Loading & Preparation**:
    * The [UCI Credit Card Default dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) is loaded.
    * The original dataset is complete, with no missing values.
    * The `ID` column is dropped, and the target variable `default.payment.next.month` is identified.

2.  **Simulating a Real-World Scenario**:
    * To test imputation methods, **Missing At Random (MAR)** data is artificially introduced into the `BILL_AMT2` and `PAY_AMT5` columns.
    * The probability of a value being missing is dependent on the `AGE` of the customer, simulating a realistic scenario where data gaps are correlated with other observed variables. Approximately 6% of the data in these two columns is made null.

3.  **Handling Missing Data: Four Strategies**:
    Four distinct datasets were created based on different handling strategies:
    * **Dataset A (Baseline): Median Imputation**: Missing values are filled with the column's median. This is a simple but common baseline.
    * **Dataset B: Linear Regression Imputation**: A linear regression model is trained on the non-missing data to predict and fill the nulls in `BILL_AMT2`. This assumes a linear relationship between the features.
    * **Dataset C: Non-Linear Regression Imputation**: A `RandomForestRegressor` is used to predict and fill the nulls in `BILL_AMT2`, capturing potentially complex, non-linear relationships.
    * **Dataset D: Listwise Deletion**: All rows containing any missing values are dropped. This is the simplest approach but results in significant data loss (over 11% of the dataset).

4.  **Model Training and Evaluation**:
    * For each of the four datasets, a **Logistic Regression** model is trained to predict credit card default.
    * The model uses `class_weight='balanced'` to handle the class imbalance in the target variable.
    * Features are standardized using `StandardScaler` before training.
    * Each model's performance is evaluated on a held-out test set using Accuracy, Precision, Recall, and F1-Score, with a focus on the "Default" class (Class 1).

---

## 📈 Performance Analysis and Results

The key findings from the model evaluations are summarized below.

| Model                     | Accuracy | F1-Score (Default) | Recall (Default) | Data Loss |
| ------------------------- | :------: | :----------------: | :--------------: | :-------: |
| **A (Median Imputation)** |  0.6793  |       0.4611       |      0.6202      |    0%     |
| **B (Linear Regression)** |  0.6792  |       0.4609       |      0.6202      |    0%     |
| **C (Non-Linear Reg)** |  0.6790  |       0.4608       |      0.6202      |    0%     |
| **D (Listwise Deletion)** |  **0.6849** |       **0.4740** |      **0.6423** |  **11.2%** |

![Model Performance Comparison Chart](model_comparison.png)

### Key Insights:
* **Similar Performance**: All imputation methods resulted in nearly identical model performance. The simple median imputation performed just as well as the more complex regression-based methods.
* **Listwise Deletion's Paradox**: Despite losing over 11% of the data, the model trained on the listwise-deleted dataset (Model D) performed **marginally better** across all key metrics.
* **The Trade-Off**: While listwise deletion shows slightly better results here, it comes at the high cost of reduced statistical power and potential selection bias, making it a risky choice in many real-world applications.
* **Linear vs. Non-Linear**: The linear regression imputation performed slightly better than the non-linear (Random Forest) one, suggesting that the underlying relationships are not complex enough to warrant a high-variance model, which might be capturing noise.

---

## 💡 Final Recommendation

Based on the analysis, the recommended strategy is:

✅ **Primary Recommendation: Linear Regression Imputation (Model B)**

### Justification:
1.  **Data Preservation**: It retains the full sample size, maximizing statistical power and ensuring that the model is trained on all available information, unlike listwise deletion.
2.  **Conceptual Soundness**: It is well-suited for the MAR (Missing At Random) data mechanism simulated in this experiment, which is a common assumption in financial datasets. It uses existing relationships between variables to make informed estimates.
3.  **Robust Performance**: It performs competitively, matching the performance of more complex methods and the simple baseline, while avoiding the pitfalls of data loss.
4.  **Practicality**: Linear regression is computationally efficient, stable, and easily interpretable, making it a reliable and practical choice for production environments.

While listwise deletion yielded the highest scores in this specific experiment, its data loss makes it less robust. **Linear Regression Imputation offers the best balance of performance, data preservation, and practical utility.**

---

