import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
drybean = pd.read_csv('Dry_Bean_Dataset.csv')
drybean

X = drybean.drop(columns=['Class'])  # Features
y = drybean['Class']  # Target variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------------------------------------------------------------
# 1st type of classification - Logistic_Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initializing the Logis regression classifier
logistic_model = LogisticRegression()

# Training the model on the training data
logistic_model.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = logistic_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing the classification report
print("Classification Report of Logistic_Regression_Classification:")
print(classification_report(y_test,y_pred))
# ------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------
# 2nd type of classification - Decison_Tree_Classification
from sklearn.tree import DecisionTreeClassifier

# Initializing the Decision Tree classifier
tree_model = DecisionTreeClassifier()

# Training the model on the training data
tree_model.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = tree_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing the classification report
print("Classification Report of Decision_Tree_Classification:")
print(classification_report(y_test,y_pred))
# ------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------
# 3rd type of classification - Random_Forest_Classification
from sklearn.ensemble import RandomForestClassifier

# Initializing the RF classifier
rf_model = RandomForestClassifier()

# Training the model on the training data
rf_model.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = rf_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing the classification report
print("Classification Report Of Random_Forest_Classification:")
print(classification_report(y_test,y_pred))
# ------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------
# 4th type of classification - Support_Vector_Machines (SVM)
from sklearn.svm import SVC

# Initializing the SVM classifier
svm_model = SVC()

# Training the model on the training data
svm_model.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = svm_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing the classification report
print("Classification Report of Support_Vector_Machine_Classification:")
print(classification_report(y_test, y_pred))
# ------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------
# 5th type of classification - k-Nearest_Neighbors (k-NN)
from sklearn.neighbors import KNeighborsClassifier

# Initializing the k-NN classifier
knn_model = KNeighborsClassifier()

# Training the model on the training data
knn_model.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = knn_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing the classification report
print("Classification Report of k-Nearest_Neighbors_Classification:")
print(classification_report(y_test, y_pred))
# -----------------------------------------------------------------------------------------------------------------------------------
