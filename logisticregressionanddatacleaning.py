import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data=pd.read_csv('F:/tequedlab/titanic.csv')
#print(data)
#sns.heatmap(data.isnull())
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1) 
#sns.heatmap(data.isnull())
data.drop('Cabin',inplace=True,axis=1) 
#sns.heatmap(data.isnull())
data.dropna(inplace=True)
#print(data)
#sns.heatmap(data.isnull())
sex=pd.get_dummies(data['Sex'],drop_first=True)
#print(sex)
embark=pd.get_dummies(data['Embarked'],drop_first=True)
#print(embark)
data.drop(['Sex','Embarked','Fare','Ticket','Name'],axis=1,inplace=True)
#print(data)
data=pd.concat([data,sex,embark],axis=1)
#print(data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.drop('Survived',axis=1),data['Survived'],test_size=.3,random_state=1)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
prediction=logreg.predict(x_test)
#print(prediction)
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,prediction)
print(cnf_matrix)
prediction1=logreg.predict([[1,3,22.0,1,0,7.25,10,1]])
#print(prediction1)
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.title("confusion matrix")
plt.xlabel("actual label")
plt.ylabel("predicted label")
print(metrics.accuracy_score(y_test,prediction))
print(metrics.precision_score(y_test,prediction))
print(metrics.recall_score(y_test,prediction))
