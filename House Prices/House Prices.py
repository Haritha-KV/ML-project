# Importing libraries
import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Creating the dataset
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [5, 10, 15, 20, 25],
    'price': [300000, 400000, 500000, 600000, 650000]
}
df = pd.DataFrame(data)

# Step 2: Features and target
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Step 3: Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Step 4: Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Making predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Check R² only if valid
if len(y_test) > 1:
    r2 = r2_score(y_test, y_pred)
    print("R² Score:", r2)
else:
    print("Not enough samples to calculate R² score.")

# Step 7: Plotting actual vs predicted
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label='Actual Price', marker='o')
plt.plot(y_pred, label='Predicted Price', marker='x')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Test Sample Index')
plt.ylabel('House Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
