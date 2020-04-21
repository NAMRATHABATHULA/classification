#importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline

#reading the data
data=pd.read_csv('mnist_train.csv')
data.head()

#extracting data from dataset
a=data.iloc[3,1:].values
#reshaping data into a reasonable size
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

#preparing data
#separating labels and the data values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)
#check data
y_train.head()

#calling rf classifier
rf=RandomForestClassifier(n_estimators=100)
#fit the model
rf.fit(x_train,y_train)
#prediction on test data
pred=rf.predict(x_test)

pred

#check prediction accuracy
s=y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
        
count

len(pred)

length=len(pred)
#accuracy
length/count
