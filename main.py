# Recurrent Neural Network
time_steps = 15
date_format = '%Y-%m-%d'

# Part 1 - Data Preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv('csv/lpz-cases.csv', header=None)
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
# normalisation is recomended for RNN instead of standarization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # all scaled prices should be between 0 and 1
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 10 timesteps and 1 output
X_train = []
y_train = []
for i in range(time_steps, len(training_set)):
    X_train.append(training_set_scaled[i-time_steps:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping
# X_train.shape[0] number of lines
# X_train.shape[1] number of columns
# 1 number of predictors
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# initialising the RNN
regressor = Sequential()

# adding the 1st LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # the first dimension (X_train.shape[0]) is not required because is automatically used
regressor.add(Dropout(0.2))

# adding the 2nd LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding the 3rd LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding the 4th LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# adding the output layer
regressor.add(Dense(units=1))
              
# compiling the RNN
# RMSprop optimizer is also recommended for RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs=2000, batch_size=32)

# Part 3 - Predict the data

from datetime import datetime, timedelta
last_date = dataset_train.iloc[-1][0]
last_date = datetime.strptime(last_date, date_format)
days = []
predicted_data = []
i = 0

# the first input should be the last values from the training set
X_last = np.array(training_set[-time_steps:], dtype='float32')

predict_range = 30
for i in range(predict_range):
    # reshape and transform the original values
    inputs = X_last.reshape(-1, 1)
    inputs = sc.transform(inputs) #  apply the same scale applied for the train set
    X_test = [inputs]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # predict the next value
    predicted_value = regressor.predict(X_test)
    predicted_value = sc.inverse_transform(predicted_value)  # inverse the scales and return the original scales
    if predicted_value[0][0] < 0:
        predicted_value[0][0] = 0

    # add the predicted value at the end and drop the first one in the X_last list
    X_last = np.concatenate((X_last[1:], predicted_value), axis=0)
    
    # visualising the results
    v = predicted_value[0]
    date = last_date + timedelta(days=i)
    days.append([date.date(), 0])
    predicted_data.append([date.date(), v[0]])
    
training_data = dataset_train.values
training_data = [[datetime.strptime(d[0], date_format), d[1]] for d in training_data]

plt.figure(figsize=(12,10))
plt.plot([p[0] for p in training_data], [p[1] for p in training_data], color='red', label='Real data')
plt.plot([p[0] for p in predicted_data], [p[1] for p in predicted_data], color='blue', label='Predicted data')
plt.title('Value prediction')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
