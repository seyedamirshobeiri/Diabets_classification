#use libarary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

#read datasets
data=pd.read_csv('diabetes.csv')

#split data
features_col=data.drop('Outcome',axis=1)
target=data['Outcome']

#split data for model
x_train,x_test,y_train,y_test=train_test_split(features_col,target,test_size=0.3,random_state=1)

#creat decisiontreeclassifier
model=DecisionTreeClassifier(criterion='entropy',max_depth=3)

#train model
model=model.fit(x_train,y_train)

#predict model
y_pred=model.predict(x_test)

#accuracy of this model
print('Accuracy is:',metrics.accuracy_score(y_test,y_pred))