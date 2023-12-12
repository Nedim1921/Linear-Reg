import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error


data = pd.read_csv('housing_price_dataset.csv')
# print(data.head())

categorical_col = ['Neighborhood'] 

data = data[['SquareFeet', 'Bedrooms', 'Neighborhood', 'Price']]

# print('*'*100)
# print(data.head())

predict = 'Price'

# Transformisanje kolone 'Neighborhood' iz stringa u dodatne kolone, koje ce imati vrijednosti 0 ili 1
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_col)], remainder='passthrough')

X = transformer.fit_transform(data.drop([predict], axis=1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = LinearRegression()

""" best = 0
for _ in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # ---- Training proccess ----
    model = LinearRegression()

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        # Saving model
        with open('housing_price.pickle', 'wb') as f:
            pickle.dump(model, f) """


pickle_in = open('housing_price.pickle', 'rb')
model = pickle.load(pickle_in)


# Predviđanja za testni skup
predictions = model.predict(X_test)

# Rezultati testiranja
results = pd.DataFrame(X_test, columns=['SquareFeet', 'Bedrooms', 'Neighborhood_Rural', 'Neighborhood_Subrub', 'Neighborhood_Urban'])

# pd.set_option('display.float_format', '{:.2f}'.format)
results['Actual Price'] = y_test
results['Predicted Price'] = predictions

print(results)


# MAE
mae = mean_absolute_error(y_test, predictions).round(2)
print(f"MAE: {mae}")

# MSE 
mse = mean_squared_error(y_test, predictions).round(2)
print(f"MSE: {mse}")


# RMSE
rmse = mean_squared_error(y_test, predictions, squared=False).round(2)
print(f"RMSE: {rmse}")


# Iscrtavanje grafikona    
# p = 'SquareFeet'
# style.use('ggplot')
# plt.scatter(data[p], data[predict])
# plt.xlabel(p)
# plt.ylabel('Price $')
# plt.show()