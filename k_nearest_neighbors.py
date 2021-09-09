import pandas as pd
import numpy as np

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names=['Sepal-length','Sepal-width','Petal-length','Petal-width','Class']
data=pd.read_csv(url,names=names
#print(data)
x=data.iloc[:,:-1].values
#print(x)
y=data.iloc[:,4].values
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=0)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)
#print(x_train)
#print(x_test)
from sklearn.neighbors import KNeighborsClassifier
Classifier=KNeighborsClassifier(n_neighbors=12)
Classifier.fit(x_train,y_train)
y_pred=Classifier.predict(x_test)
print(y_pred)

from sklearn import metrics
c=confusion_matrix(y_test,y_pred)

import seaborn as sns
sns.heatmap(pd.DataFrame(c),annot=True,fmt='g')
print(metrics.accuracy_score(y_test,y_pred))
