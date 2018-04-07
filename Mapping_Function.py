import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#100 evenly spaced spred of numbers between 0, 2pi
x = np.linspace(0, 2*np.pi, 100)

#print(x)
#random generated seed
np.random.seed(321)
#100 random numbers in bell curve .5 std dev
noise = np.random.normal(0, .5, 100)

#mapping function + noise
y = np.sin(x) + noise
#data frame aka table with x and y as columns/rows using dictionary
df = pd.DataFrame({"input":x, "target":y})
print(df.head(10))

#scatter plot of noise and line of sin
print(plt.scatter(df.input, df.target))
plt.plot(df.input, np.sin(df.input), color='r')
plt.show()

######Mean Model
mean_model = np.mean(df.target)
print(mean_model)

plt.scatter(df.input, df.target)
plt.plot(df.input, ([mean_model]*len(df.input)),'r+' )
plt.show()

#####Linear Regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

features = df.drop('target', axis=1)
target = df.target

lr.fit(features, target)
#y=mx + b
print(lr.intercept_)   #y-intercept
print(lr.coef_)        #slope

plt.scatter(df.input, df.target)
plt.plot(df.input, lr.predict(features), 'k--')
plt.plot(df.input, ([mean_model]*len(df.input)),'r+' )
plt.show()



