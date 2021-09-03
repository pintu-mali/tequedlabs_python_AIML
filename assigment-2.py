import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset=pd.read_csv('F:/tequedlab/covid-19.csv')
#print(dataset)
x=dataset[['fever','bodyPain','age','runnyNose','diffBreath']]
#print(x)
y=dataset[['infectionProb']]
#print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=0)
#print(x_train)
#print(x_test)
#print(y_test)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
Y_pred=logreg.predict(x_test)
#print(pd.DataFrame(y_test,Y_pred))
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,Y_pred)
#print(cnf_matrix)

import seaborn as sns
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.title("COVID-19")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
print(metrics.accuracy_score(y_test,Y_pred))
print(metrics.precision_score(y_test,Y_pred))
print(metrics.recall_score(y_test,Y_pred))
#Y_pred1=logreg.predict([[90,0,24,1,-1]])
# print(Y_pred1)
