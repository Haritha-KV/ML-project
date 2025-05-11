import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 📊 Updated sample dataset (10 data points)
data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 2000, 2500, 3200, 2800, 2200],
    'Bedrooms': [3, 4, 3, 5, 4, 3, 4, 5, 4, 3],
    'Bathrooms': [2, 3, 2, 4, 3, 2, 3, 4, 3, 2],
    'Location_Score': [7, 8, 9, 6, 7, 8, 7, 5, 6, 8],
    'Price': [300000, 400000, 450000, 600000, 550000, 380000, 470000, 620000, 530000, 410000]
}

# Create DataFrame
df = pd.DataFrame(data)

# ✅ Features and target
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Location_Score']]
y = df['Price']

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Now test set will have 3 samples

# 🧠 Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Predict
y_pred = model.predict(X_test)

# 📊 Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🔍 Model Evaluation")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print("\nPredicted Prices:", y_pred)

# 📉 Plot actual vs predicted
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()
