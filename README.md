# ITP449Final
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read the dataset into a dataframe
data = pd.read_csv('jobs_change.csv')
display(data.head())
data.shape

# Check for missing values
data.isna().any()
data.replace('NaN', 'Unknown', inplace=True)
# Plot the graph of gender against job change
plt.figure()
gender_plot = sns.countplot(x= data['gender'], hue= 'target', data= data)
# Plot the graph of city against job change
plt.figure()
company_plot = sns.countplot(x= data['city'], hue= 'target', data= data)
# Plot the graph of experience against job change
plt.figure()
experience_plot = sns.countplot(x= data['relevent_experience'], hue= 'target', data= data)
# Plot the graph of education level against job change
plt.figure()
education_plot = sns.countplot(x= data['education_level'], hue= 'target', data= data)
# Plot the graph of major against job change
plt.figure()
major_plot = sns.countplot(x= data['major_discipline'], hue= 'target', data= data)
# Plot the graph of company size against job change
plt.figure()
company_plot = sns.countplot(x= data['company_size'], hue= 'target', data= data)
# Plot the graph of company type against job change
plt.figure()
company_plot = sns.countplot(x= data['company_type'], hue= 'target', data= data)

# Drop the irrelevant features
data.drop(['enrollee_id', 'city', 'company_size', 'company_type'],axis = 1, inplace = True)
# Convert 'experience' and 'last_new_job' into categorical variables
data['experience'].replace({'<1':0,'>20':21, 'Unknown':-1}).astype('float')
data['last_new_job'].replace({'never':0,'>4':5, 'Unknown':-1}).astype('float')
# Convert categorical variables into dummy variables
data = pd.get_dummies(data, columns=['gender','enrolled_university','relevent_experience','education_level','major_discipline', 'experience','last_new_job'])
pd.set_option('display.max_columns', None)
print(data.head())

# Split into features and targets using iloc to select columns
X = data.drop("target",axis=1)
y = data["target"]
print(y)

# Partition the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Get the baseline accuracy 
from sklearn.dummy import DummyClassifier
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_train,y_train)
baseline_acc = dummy_classifier.score(X_test,y_test)
print("Baseline Accuracy = ", baseline_acc)

# Fit the training data to a classification tree model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
training_pred = dt.predict(X_train)
acc_training = accuracy_score(training_pred, y_train)
print('Classification Tree Accuracy=', acc_training)

# Fit the training data to a bagging model
from sklearn.ensemble import BaggingClassifier
model_bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)
model_bagging.fit(X_train, y_train)
pred_bagging = model_bagging.predict(X_test)
acc_bagging = accuracy_score(y_test, pred_bagging)
print('Bagging Accuracy=', acc_bagging)

# Fit the training data to a random forest model
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, max_features=5)
model_rf.fit(X_train, y_train)
predict_rf = model_rf.predict(X_test)
print('Random Forest Accuracy=', accuracy_score(y_test, predict_rf))

# Fit the training data to an adaboost model
from sklearn.ensemble import AdaBoostClassifier
base_est = DecisionTreeClassifier(max_depth=5)
ada_boost = AdaBoostClassifier(base_est, n_estimators=100, learning_rate=.05)
ada_boost.fit(X_train, y_train)
print('Adaboost Accuracy=', accuracy_score(y_test, ada_boost.predict(X_test)))

# Fit the training data to a voting model
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
rfClf = RandomForestClassifier(n_estimators=100) 
svmClf = SVC(probability=True) 
logClf = LogisticRegression()
dtClf = DecisionTreeClassifier(max_depth=5)
clf = VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf), ('log', logClf), ('dtClf', dtClf)], voting='soft') 
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
print('Voting Accuracy=', accuracy_score(y_test, clf_pred))

# conclude on the best fitting model
print('Since Classification Tree Model has the highest accuracy score, it is the most promising model.')
