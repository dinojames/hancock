import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = pd.read_csv('linear_reg_data.csv')
data.drop(['Unnamed: 0'], axis=1)

x = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(x, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

predictions = reg.predict(x)
plt.figure(figsize=(8, 5))
plt.scatter(
    data['TV'],
    data['sales'],
    c='black'
)
plt.plot(
    data['TV'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()