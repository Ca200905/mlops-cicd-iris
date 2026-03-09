from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# load dataset
X, y = load_iris(return_X_y=True)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# save model
os.makedirs("app", exist_ok=True)

with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
