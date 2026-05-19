import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("train.csv")

# Select columns
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Target column
y = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))

# User input
area = float(input("Enter square footage: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))

new_house = np.array([[area, bedrooms, bathrooms]])

# Predict price
predicted_price = model.predict(new_house)

print("Predicted House Price: $", round(predicted_price[0], 2))