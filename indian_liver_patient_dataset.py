import numpy as np
import pandas as pd

dataset=pd.read_csv('indian_liver_patient_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_gender=LabelEncoder()
dataset['gender']=lbl_gender.fit_transform(dataset['gender'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
'''
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

from sklearn.decomposition import PCA 
pca=PCA(n_components=2)
pca.fit(x)
x=pca.transform(x)
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=8)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))
