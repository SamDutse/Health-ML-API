from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import joblib

# ========== DIABETES (Regression) ==========
diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

diabetes_model = RandomForestRegressor(n_estimators=100, random_state=42)
diabetes_model.fit(X_train_d, y_train_d)

preds_d = diabetes_model.predict(X_test_d)
mse = mean_squared_error(y_test_d, preds_d)
mae = mean_absolute_error(y_test_d, preds_d)

print(f"Diabetes model MAE: {mae:.2f}")
print(f"Diabetes model MSE: {mse:.2f}")

joblib.dump(diabetes_model, "diabetes_model.pkl")
print("Saved diabetes_model.pkl")


# ========== BREAST CANCER (Classification) ==========
cancer = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42
)

cancer_model = RandomForestClassifier(n_estimators=100, random_state=42)
cancer_model.fit(X_train_c, y_train_c)

preds_c = cancer_model.predict(X_test_c)
acc = accuracy_score(y_test_c, preds_c)

print(f"Breast cancer model accuracy: {acc:.2f}")

joblib.dump(cancer_model, "cancer_model.pkl")
print("Saved cancer_model.pkl")
