# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

# Creating a sample dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7],
    'Age': [22, 25, 28, 30, 32, 36, 40],
    'Salary': [25000, 30000, 35000, 40000, 45000, 50000, 60000]
}

df = pd.DataFrame(data)

# ============================
# 1. Simple Linear Regression (Experience -> Salary)
# ============================

# Features and Target
X_simple = df[['Experience']]
y = df['Salary']

# Train-Test Split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.3, random_state=0)

# Model Training
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

# Prediction
y_pred_s = model_simple.predict(X_test_s)

# Evaluation
r2_simple = r2_score(y_test_s, y_pred_s)

print("\n📈 Simple Linear Regression (Experience -> Salary)")
print(f"Coefficient: {model_simple.coef_[0]}")
print(f"Intercept: {model_simple.intercept_}")
print(f"R² Score: {r2_simple:.2f}")

# Plotting Simple Regression Line
plt.figure(figsize=(8,5))
plt.scatter(X_simple, y, color='red', label='Actual Salary')
plt.plot(X_simple, model_simple.predict(X_simple), color='blue', label='Regression Line')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# ============================
# 2. Multiple Linear Regression (Experience + Age -> Salary)
# ============================

# Features and Target
X_multi = df[['Experience', 'Age']]
y_multi = df['Salary']

# Train-Test Split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.3, random_state=0)

# Model Training
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

# Prediction
y_pred_m = model_multi.predict(X_test_m)

# Evaluation
r2_multi = r2_score(y_test_m, y_pred_m)

print("\n🧮 Multiple Linear Regression (Experience + Age -> Salary)")
print(f"Coefficients: {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_}")
print(f"R² Score: {r2_multi:.2f}")

# ========== 3D Graph for Multiple Linear Regression ==========

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual points
ax.scatter(df['Experience'], df['Age'], df['Salary'], color='red', label='Actual Points')

# Create grid to plot the plane
xp, yp = np.meshgrid(
    np.linspace(df['Experience'].min(), df['Experience'].max(), 10),
    np.linspace(df['Age'].min(), df['Age'].max(), 10)
)

# Predict Z (Salary) values
zp = model_multi.intercept_ + model_multi.coef_[0]*xp + model_multi.coef_[1]*yp

# Plot regression plane
ax.plot_surface(xp, yp, zp, alpha=0.5, color='blue')

# Labels
ax.set_xlabel('Experience')
ax.set_ylabel('Age')
ax.set_zlabel('Salary')
ax.set_title('Multiple Linear Regression (3D)')

plt.legend()
plt.show()
