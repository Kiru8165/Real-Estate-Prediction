import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('Bengaluru_House_Data.csv')
print(df.head())
df = df.dropna()  # Drop missing values
df = pd.get_dummies(df, drop_first=True)  # Convert categorical columns

X = df.drop('price', axis=1)  # Replace 'price' with your target column
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'real_estate_model.pkl')

