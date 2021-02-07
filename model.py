import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import initializers

df = pd.read_csv("revisi.csv", delimiter=",")

df.head()

dataset = df.values

X = dataset[:,0:6]
y = dataset[:,6]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x_scaler = min_max_scaler.fit_transform(X)


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(x_scaler, y, test_size=0.2)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


model = Sequential([
    Dense(10,activation='sigmoid', input_shape=(6,)),
    Dense(1,activation='sigmoid'),
])
model.summary()

model.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss = 'mse', 
    metrics = ['accuracy'])

history = model.fit(X_train, Y_train,batch_size=32, epochs=100,validation_data=(X_val, Y_val))

res_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))

model.predict([[2,3,1,0,3,1]])

model.save("model.h5")
print("Saved model to disk")
