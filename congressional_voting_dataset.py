import numpy as np
import pandas as pd

dataset=pd.read_csv('congressional_voting_dataset.csv')
dataset.replace(to_replace ='y',value=int(1),inplace=True)
dataset.replace(to_replace='n',value=int(0),inplace=True)
dataset.replace(to_replace='?',value=int(1000000),inplace=True)


from sklearn.preprocessing import LabelEncoder
lbl_party=LabelEncoder()
dataset['political_party']=lbl_party.fit_transform(dataset['political_party'])

x=pd.DataFrame(dataset.iloc[:,:-1].values)
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)

print(confusion)
print(model.score(x_test,y_test))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title('Classification')
plt.xlabel('VoterId')
plt.ylabel('political_party')
plt.scatter(range(0,109),y_predict)
plt.show()