#Only for practice

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns #for projeccting heat map if required

data=pd.read_csv("F:/tequedlab/salary_data.csv")
#print(dataset)

x=data[['YearsExperience']]
#print(x)
y=data[['Salary']]

plt.plot(x,y,label="Experience vs Salary",c='r')
plt.title("E VS S")
plt.xlabel("Expirience")
plt.ylabel("Salary")
plt.legend()
plt.grid()
plt.show()

x=x.iloc[:20]
#print(x)
y=y.iloc[:20]
#print(y)

plt.scatter(x,y,label="e vs s",marker='*',c='r')
plt.title("E VS S")
plt.xlabel("Expirience")
plt.ylabel("Salary")
plt.legend()
plt.grid()
plt.show()


y=[1,2,3,4]
x=[10000,20000,30000,40000]
x1=[1.5,2.5,3.5,4.5]
y1=[1000,20023,45822,33467]
plt.bar(x,y,label='bmw',width=600,color='r')
plt.barh(x1,y1,label="cisco",height=.1,color='b')
plt.title("E VS S")
plt.xlabel("Salary")
plt.ylabel("Expirience")
plt.legend()
plt.show()

JOBS_AVAILABLE=[20,33,35,38,22,28,22,20,27,80,85,85,49,22,49]
age_range=[0,10,20,30,40,50,100]
plt.hist(population,age_range,label="p vs a",histtype='stepfilled',rwidth=.1,color='y')
plt.title("J VS A")
plt.ylabel("JOBS")
plt.xlabel("age")
plt.legend()
plt.show()

to_do=['GYM','GET READY','WORK','STUDY','timepass','GO HOME','SLEEP']
duration=[1,2,6,6,3,1,5]
plt.pie(duration,labels=to_do,shadow=True,autopct="%1.2f%%")
plt.title("DAILY ACTIVITIES")
plt.show()


