import numpy as np
import pandas as pd

dataset=pd.read_csv('mammogram_dataset.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values



from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(x)
x=pca.transform(x)

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))