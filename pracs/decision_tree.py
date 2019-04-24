import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#Importing dataset
dataset = pd.read_csv('decision_tree_data.csv')

#Splitting the dataset
x = dataset.values[:, 0:4]
y = dataset.values[:, 4]

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.3)

#Gini Index
clf_gini = DecisionTreeClassifier()
clf_gini.fit(x_train, y_train)

#Information Gain
'''clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(x_train, y_train)

#Predicting the branches on test data'''
print(x_test)
y_pred = clf_gini.predict(x_test)
print(y_pred)

#Branching accuracy
print('Accuracy is ' +str(accuracy_score(y_test, y_pred)*100)+ '%')
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 