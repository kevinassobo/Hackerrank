# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
import pandas as pd
from sklearn import linear_model as  lm

N = int(input())
y_train = np.array([input().split()[1] for _ in range(N)], int)
X_train = np.array(range(1,N+1))
X_train = X_train.reshape(-1,1)
X_test = np.array(range(N+1,N+13))
X_test = X_test.reshape(-1,1)

model = lm.LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print('\n'.join(list(map(str,predictions))))
