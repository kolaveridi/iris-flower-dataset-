# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 01:19:04 2017

@author: User
"""
#import all that is required
import pandas as pd
import numpy as np
import seaborn as sns #Seaborn will be used for making visualizations.
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression  #baba

#read the file
dataset = pd.read_csv('iris.csv')
dataset = dataset[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]]  
#do the visualization
sns.lmplot("Sepal.Length", "Sepal.Width", data=dataset, hue="Species", fit_reg=False)
#again visualization
sns.lmplot("Petal.Length", "Petal.Width", data=dataset, hue="Species", fit_reg=False)  

#again visulaization
sns.pairplot(data=dataset, hue="Species")
X = np.array(dataset.drop(["Species"],1))  #https://chrisalbon.com/python/pandas_dropping_column_and_rows.html
y = np.array(dataset["Species"])  
#trainig and testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#define calssifier
classifier = LogisticRegression()
#fitting the training
classifier.fit(X_train, y_train)
#score function on our testing data.
conf = classifier.score(X_test, y_test) 
#printing part 
print(conf)

