import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('../data/sampregdata.csv')
data.drop('Unnamed: 0', axis=1, inplace=True)
print(data.corr())
# x4 has highest correlation (-0.524) followed by x2 (-0.471)

# 1b)
x = data[['x4']]
y = data['y']

# Train and save model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f"MSE_1 = {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²_1 = {r2_score(y_test, y_pred):.2f}")
# MSE_1 = 85.79
# R²_1 = 0.26
joblib.dump(model, '../models/model_v1.pkl')
x_train.to_csv("../data/x_train1.csv", index=False)
x_test.to_csv("../data/x_test1.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)


