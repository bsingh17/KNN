import numpy as np
import pandas as pd

dataset=pd.read_csv('balance_scale_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_balance=LabelEncoder()
dataset['balance']=lbl_balance.fit_transform(dataset['balance'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=20)
model.fit(x_train,y_train)

y_predict=model.predict(x_test)
y_predict=pd.DataFrame(y_predict)
dataset['predicted']=y_predict

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))

import matplotlib.pyplot as plt
plt.title('Classification')
plt.xlabel('left_weight')
plt.ylabel('balance')
plt.scatter(range(0,94) ,y_predict,c='green')
plt.show()
