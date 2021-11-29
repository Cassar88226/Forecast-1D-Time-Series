import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional, TimeDistributed
from sklearn.metrics import mean_squared_error
from math import sqrt

# past steps
n_steps = 2000
# train test split percent
train_percent = 0.9

# build a univariate sequence
def build_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_index = i + n_steps
        # check if end index is over sequence
        if end_index + 1 > len(sequence):
            break
        # get input and output
        seq_x, seq_y = sequence[i:end_index], sequence[end_index]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def split_series(series, n_past, n_future):
    #
    # n_past ==> no of past observations
    #
    # n_future ==> no of future observations 
    #
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end], series[past_end:future_end]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

df = pd.read_csv("dataset_GBPUSD_EURUSD.csv")
spread = list(df["SPREAD"])

plt.figure(figsize=(20,10))
plt.plot(spread)
plt.show()

train_size = int((len(spread) - n_steps) * train_percent)
train_dataset = spread[n_steps:train_size+n_steps]
test_dataset = spread[train_size + n_steps:]

plt.figure(figsize=(20,10))
plt.plot(train_dataset)
plt.plot([None for i in train_dataset] + [x for x in test_dataset])
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

X, y = build_sequence(spread, n_steps)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

train_size = int(len(X) * train_percent)
X_train, y_train = X[0:train_size], y[0:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# define model
model = Sequential()
model.add(Bidirectional(LSTM(512, activation='tanh', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
history = model.fit(X_train, y_train, epochs=20)

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("model loss.png")

model.save("BI-LSTM epoch-20.h5")

predictions = []
for x_input in X_test:
    x_input = x_input.reshape((1, n_steps, n_features))
#     print(x_input)
    yhat = model.predict(x_input)
#     print(yhat)
    predictions.append(yhat[0][0])

plt.figure(figsize=(20,10))
plt.plot(train_dataset)
plt.plot([None for i in train_dataset] + [x for x in test_dataset])
plt.plot([None for i in train_dataset] + [x for x in predictions])
plt.legend(['Train', 'Test', 'Predicted'], loc='upper left')
plt.savefig("predicted vs train vs test.png")

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    return mean_squared_error(actual, predicted , squared=True)

RMSE = evaluate_forecasts(test_dataset, predictions)
print(RMSE)


prediction_steps = 2016
# demonstrate prediction
forecasted = []

x_input = X_test[-1:]
x_input = np.delete(x_input, 0)
x_input = np.append(x_input, spread[-1])
x_input = x_input.reshape((1, n_steps, n_features))
# print(x_input)

for i in range(prediction_steps):
    yhat = model.predict(x_input)
    forecasted.append(yhat[0][0])
#     print(yhat)
    x_input = np.delete(x_input, 0)    
    x_input = np.append(x_input, yhat[0][0])
    
    x_input = x_input.reshape((1, n_steps, n_features))
#     print(x_input)

plt.figure(figsize=(20,10))
plt.plot(spread)
plt.plot([None for i in spread] + [x for x in forecasted])
plt.legend(['Past', 'Predicted'], loc='upper left')
plt.savefig("forecasted.png")