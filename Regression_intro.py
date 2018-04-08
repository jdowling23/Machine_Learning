import pandas as pd
import quandl as q, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df = q.get("WIKI/GOOGL")
#print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'] * 100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_CHANGE', 'Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#days ahead for prediction
forecast_out = int(math.ceil(0.01*len(df)))
print('forecast out = ' + str(forecast_out))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
x = preprocessing.scale(x)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression()
#clf = svm.SVR()    #Support vector regression
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print('Accuracy = ' + str(accuracy))
