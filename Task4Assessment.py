import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset from a CSV file
file_path = 'updated_NSCH_Vision_Health_Data.csv'
df = pd.read_csv(file_path)

# Set the target variable as 'Data_Value'
target_column_name = 'Data_Value'

# Feature and Target
X = df.drop(target_column_name, axis=1)
y = df[target_column_name]

# Convert categorical variables to dummy/indicator variables if necessary
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **1. Linear Regression (Regression - Predicting continuous values)**
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
linear_pred = linear_model.predict(X_test_scaled)
print(f"\nLinear Regression R^2: {r2_score(y_test, linear_pred)}")

# **2. Decision Tree Regressor (Regression)**
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train_scaled, y_train)

# Predict and Evaluate
dt_regressor_pred = dt_regressor.predict(X_test_scaled)
print(f"\nDecision Tree Regressor R^2: {r2_score(y_test, dt_regressor_pred)}")

# **3. Random Forest Regressor (Regression)**
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Predict and Evaluate
rf_regressor_pred = rf_regressor.predict(X_test_scaled)
print(f"\nRandom Forest Regressor R^2: {r2_score(y_test, rf_regressor_pred)}")

# **4. Support Vector Regression (SVR) (Regression)**
svr_model = SVR(kernel='linear')
svr_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
svr_pred = svr_model.predict(X_test_scaled)
print(f"\nSVR R^2: {r2_score(y_test, svr_pred)}")

# **5. K-Means Clustering (Unsupervised - to cluster data)**
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)

# Print cluster centers
print(f"\nK-Means Cluster Centers: {kmeans.cluster_centers_}")

# Predict clusters
kmeans_pred = kmeans.predict(X_test_scaled)
print(f"Clusters predicted by KMeans: {np.unique(kmeans_pred)}")

# For classification, we need to convert the target variable to categorical
# Let's assume we want to classify values above the median as 1, and below as 0
y_class = (y > y.median()).astype(int)

# Split data for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Standardize the features for classification
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

# **Classification Models**

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train_class_scaled, y_train_class)
dt_pred = dt_clf.predict(X_test_class_scaled)

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_class_scaled, y_train_class)
rf_pred = rf_clf.predict(X_test_class_scaled)

# Support Vector Machine (SVM) Classifier
svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_class_scaled, y_train_class)
svm_pred = svm_clf.predict(X_test_class_scaled)

# K-Nearest Neighbors (KNN) Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_class_scaled, y_train_class)
knn_pred = knn_clf.predict(X_test_class_scaled)

# Naive Bayes Classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train_class_scaled, y_train_class)
nb_pred = nb_clf.predict(X_test_class_scaled)

# **Model Evaluation Summary:**
print("\nModel Evaluation Summary:")
# **Classification Models (Accuracy)**
print(f"\nAccuracy (Decision Tree): {accuracy_score(y_test_class, dt_pred)}")
print(f"Accuracy (Random Forest): {accuracy_score(y_test_class, rf_pred)}")
print(f"Accuracy (SVM): {accuracy_score(y_test_class, svm_pred)}")
print(f"Accuracy (KNN): {accuracy_score(y_test_class, knn_pred)}")
print(f"Accuracy (Naive Bayes): {accuracy_score(y_test_class, nb_pred)}")

# **Regression Models (R^2)**
print(f"\nR^2 (Linear Regression): {r2_score(y_test, linear_pred)}")
print(f"R^2 (Decision Tree Regressor): {r2_score(y_test, dt_regressor_pred)}")
print(f"R^2 (Random Forest Regressor): {r2_score(y_test, rf_regressor_pred)}")
print(f"R^2 (SVR): {r2_score(y_test, svr_pred)}")