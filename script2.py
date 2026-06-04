import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

sns.set_theme(style="whitegrid")

data_url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(data_url)
df.columns = df.columns.str.strip()

print(f"Dataset Successfully Loaded. Initial Shape: {df.shape}")

df_lr = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)
df_tree = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=False, dtype=int)

X_lr = df_lr.drop(columns=['charges'])
y_lr = df_lr['charges']
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

for col in X_train_lr.select_dtypes(include=[np.number]).columns:
    median_val = X_train_lr[col].median()
    X_train_lr[col] = X_train_lr[col].fillna(median_val)
    X_test_lr[col] = X_test_lr[col].fillna(median_val)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
lr_pred = lr_model.predict(X_test_lr)

lr_r2 = r2_score(y_test_lr, lr_pred)
lr_mae = mean_absolute_error(y_test_lr, lr_pred)

print("\n--- OLS Linear Regression Evaluation ---")
print(f"R-squared (Variance Explained): {lr_r2:.4f} ({lr_r2 * 100:.2f}%)")
print(f"Mean Absolute Error (MAE): ${lr_mae:.2f}")

X_tree = df_tree.drop(columns=['charges'])
y_tree = df_tree['charges']
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)

for col in X_train_tree.select_dtypes(include=[np.number]).columns:
    median_val = X_train_tree[col].median()
    X_train_tree[col] = X_train_tree[col].fillna(median_val)
    X_test_tree[col] = X_test_tree[col].fillna(median_val)

tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train_tree, y_train_tree)
tree_pred = tree_model.predict(X_test_tree)

tree_r2 = r2_score(y_test_tree, tree_pred)
tree_mae = mean_absolute_error(y_test_tree, tree_pred)

print("\n--- Decision Tree Evaluation ---")
print(f"R-squared (Variance Explained): {tree_r2:.4f} ({tree_r2 * 100:.2f}%)")
print(f"Mean Absolute Error (MAE): ${tree_mae:.2f}")

cv_scores_tree = cross_val_score(tree_model, X_tree, y_tree, cv=5, scoring='r2')
cv_scores_lr = cross_val_score(lr_model, X_lr, y_lr, cv=5, scoring='r2')
print(f"\n--- Academic Robustness Check (5-Fold Cross-Validation) ---")
print(f"Linear Regression Mean R2: {cv_scores_lr.mean() * 100:.2f}%")
print(f"Decision Tree Mean R2: {cv_scores_tree.mean() * 100:.2f}%")

importances = tree_model.feature_importances_
indices = np.argsort(importances)[::-1]

policy_labels_dict = {
    'smoker_yes': 'Tobacco Use: Smoker',
    'smoker_no': 'Tobacco Use: Non-Smoker',
    'bmi': 'Body Mass Index (BMI)',
    'age': 'Age of Beneficiary',
    'children': 'Number of Dependents',
    'region_northwest': 'Region: Northwest',
    'region_southeast': 'Region: Southeast',
    'region_southwest': 'Region: Southwest',
    'region_northeast': 'Region: Northeast',
    'sex_male': 'Sex: Male',
    'sex_female': 'Sex: Female'
}

all_clean_labels = [policy_labels_dict.get(col, col) for col in X_tree.columns[indices]]
all_importances = importances[indices]

plot_indices = all_importances > 0.001
plot_labels = [all_clean_labels[i] for i, keep in enumerate(plot_indices) if keep]
plot_importances = all_importances[plot_indices]

print("\n--- Policy Insights: Key Drivers of Healthcare Expenditure ---")
for f in range(len(plot_labels)):
    print(f"{f + 1}. {plot_labels[f]}: {plot_importances[f]*100:.2f}%")

plt.figure(figsize=(10, 6))
sns.barplot(
    x=plot_importances,
    y=plot_labels,
    hue=plot_labels,
    palette="viridis",
    legend=False
)
plt.title("Feature Importance: Core Drivers of Public Healthcare Costs", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Relative Importance Score", fontsize=12)
plt.ylabel("Risk Factors & Demographics", fontsize=12)

plt.figtext(0.15, -0.02, "Note: Modifiable behavioral risks (tobacco use) drastically outweigh static demographics in fiscal impact.",
            fontsize=10, style='italic', color='dimgray')
plt.tight_layout()

plt.savefig("healthcare_feature_importance.png", bbox_inches='tight', dpi=300)
plt.show()

print("\n[Graphics Generated] 'healthcare_feature_importance.png' has been saved successfully.")

print("\n--- FINAL POLICY CONCLUSION ---")
if cv_scores_tree.mean() > cv_scores_lr.mean():
    print(f"Winner Model: Decision Tree Regressor (Outperformed Linear Regression by {(cv_scores_tree.mean() - cv_scores_lr.mean())*100:.2f}% in CV)")
    print("Policy Implication: The interaction between risk factors is non-linear. Simple linear tracking underestimates high-risk clusters.")
else:
    print("Winner Model: Linear Regression")
    print("Policy Implication: Constant marginal effects dominate; linear resource allocation is optimal.")