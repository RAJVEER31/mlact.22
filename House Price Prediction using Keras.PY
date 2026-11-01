import pandas as pd
import numpy as np
from keras.models import Sequential  # type: ignore[reportMissingTypeStubs]
from keras.layers import Dense  # type: ignore[reportMissingTypeStubs]
from keras.callbacks import History  # type: ignore[reportMissingTypeStubs]

# load dataset
df = pd.read_csv("housing_data.csv")
df.head()

# split into input (X) and output (Y) variables
y = df.pop('AboveMedianPrice')
x = df

from typing import Any, cast
from sklearn.model_selection import train_test_split  # type: ignore[reportUnknownVariableType]

X_train, X_test, y_train, y_test = cast(
	tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
	train_test_split(x, y, test_size=0.2, random_state=0),
)
X_train = X_train
X_test = X_test
y_train = y_train

# create model
model = Sequential()
model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))  # type: ignore[reportUnknownMemberType]
model.add(Dense(6, kernel_initializer='normal', activation='relu'))  # type: ignore[reportUnknownMemberType]
model.add(Dense(1, kernel_initializer='normal'))  # type: ignore[reportUnknownMemberType]
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')  # type: ignore[reportUnknownMemberType]

# Fitting the ANN to the Training set
model_history: History = cast(
	History, cast(Any, model).fit(X_train, y_train, batch_size=10, epochs=100)
)

# list all data in history
cast(Any, model).summary()
#total params are total number of weights and biases

# Predict the Test set results
Y_pred = cast(np.ndarray, cast(Any, model).predict(X_test))
print(Y_pred)

# Calculate the Model Performance
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, Y_pred)

print(mae)
