# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the training data
F,N = map(int,input().split())
train = np.array([input().split() for _ in range(N)],float)
X_train = train[:,0:F]
y_train = train[:,-1]

# Read the test data
T = int(input())
X_test = np.array([input().split() for _ in range(T)],float)

# Model
model = LinearRegression()
model.fit(X_train,y_train)

# Predictions
predictions = model.predict(X_test)

# Output
print('\n'.join(list(map(str,predictions))))
