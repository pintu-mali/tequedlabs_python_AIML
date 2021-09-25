import pandas as pd                                   #to fetch the data
import matplotlib.pyplot as plt                       #for plotting graphs
import numpy as py                                    #to store data in array
import seaborn as sns                                 #for data cleaning

#importing the data

data=pd.read_csv("F:/tequedlab/Mall_Customers.csv")
#print(data)

#data cleaning

sns.heatmap(data.isnull())                            #By this we got to know that our data set has no empty data
plt.show()

data.describe()                                       #to get detailed information about the data

#plotting graphs


sns.countplot(y='Gender',data=data)                   # by this we got to know that in our data we have more females


sns.relplot(y='Annual Income (k$)',x='Age',kind='line',data=data,height=4,aspect=2) #to check which age ppl have more annual income


sns.catplot(x='Gender',y='Spending Score (1-100)',kind='strip',data=data)           #to compare their spending score


sns.catplot(x='Gender',y='Annual Income (k$)',kind='violin',data=data)              # we are plotting violin to check which gender has more annual income and  where they have more majority 
plt.show()





#forming cluster using Kmeans clustering


#converting string data into integer

sex=pd.get_dummies(data['Gender'],drop_first=True)
#print(sex)

#dropping the unnecessary data

data.drop(['CustomerID','Gender'],axis=1,inplace=True) # so here customer id are not import in our prediction so we are dropping it
data =pd.concat([data,sex],axis=1)
print(data)
X=data.iloc[:,[1,2,3]].values                          #takin gender , Annual Income , Spending Score in x
print(X)

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):                                  #using elbow method to know how many clusters to form
    kmeans= KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('k')
plt.ylabel('var')
plt.show()
                                    

kmeans= KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)   #so here we are using 5 clusters which we got through  elbow method
Y_Kmeans=kmeans.fit_predict(X)
print(Y_Kmeans)


plt.scatter(X[Y_Kmeans==0,0],X[Y_Kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[Y_Kmeans==1,0],X[Y_Kmeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[Y_Kmeans==2,0],X[Y_Kmeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[Y_Kmeans==3,0],X[Y_Kmeans==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[Y_Kmeans==4,0],X[Y_Kmeans==4,1],s=100,c='black',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.legend()
plt.show()







