from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

model = LinearRegression()
model.fit(X, y)

print("Food Demand Prediction Model Trained!")

prediction = model.predict([[6]])

print("Predicted Food Demand:", prediction[0])
