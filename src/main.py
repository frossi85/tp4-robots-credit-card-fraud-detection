import pandas as pd
import numpy as np 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import zipfile

# extranct data file
zip_ref = zipfile.ZipFile("../data/creditcard.csv.zip", 'r')
zip_ref.extractall("../data")
zip_ref.close()

# Load the dataset
data_frame = pd.read_csv("../data/creditcard.csv")

# See class distribution in the data frame
count_classes = pd.value_counts(data_frame['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

plt.show()

# Rescale Time and Amount. The amount and time columns are not in line with the anonimised features.
data_frame['normAmount'] = StandardScaler().fit_transform(data_frame['Amount'].reshape(-1, 1))
data_frame['normTime'] = StandardScaler().fit_transform(data_frame['Time'].reshape(-1, 1))

#Dropping the old Time and Amount columns
data_frame = data_frame.drop(['Time','Amount'], axis = 1)   

entire_data_frame = data_frame

#Undersampling
fraud_count = len(data_frame[data_frame.Class == 1])
fraud_index = data_frame[data_frame.Class == 1].index
non_fraud_index = data_frame[data_frame.Class == 0].index
random_sample_index = np.random.choice(non_fraud_index, fraud_count, replace = False)
random_sample_index = np.array(random_sample_index)
under_sample_index = np.concatenate([random_sample_index,fraud_index])
total_undersample_dataset = data_frame.iloc[under_sample_index,:]
y = total_undersample_dataset.Class
X = total_undersample_dataset.drop('Class', 1)

# Undersampled dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state = 42)

print("")
print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train) + len(X_test))

# Showing ratio
print("Percentage of normal transactions: ", len(total_undersample_dataset[total_undersample_dataset.Class == 0]) / len(total_undersample_dataset))
print("Percentage of fraud transactions: ", len(total_undersample_dataset[total_undersample_dataset.Class == 1]) / len(total_undersample_dataset))
print("Total number of transactions in resampled data: ", len(total_undersample_dataset))

print("")
print("")
print("##### KNeighbors Classifier")

# Hyperparameter Tuning for K
K_list = list(range(1, 100))

#create empty list
cv_scores = []

#perform K search
for k in K_list:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
    

# Plotting misclassification error
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = K_list[MSE.index(min(MSE))]
print("")
print("The optimal number of neighbors is %d" % optimal_k)

# Try KNN to see how well it predicts fraud on the undersampled dataset
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(X_train, y_train)
y_knn_predict = knn.predict(X_test)

print("")
print("Classification Report for test records")
print("")
print(classification_report(y_test, y_knn_predict))

#applying the model trained on undersampled data to the entire dataset
y_entire = entire_data_frame.Class
X_entire = entire_data_frame.drop('Class', 1)
X_train_entire, X_test_entire, y_train_entire, y_test_entire = train_test_split(X_entire, y_entire, test_size = .2, random_state = 42)
predict_y = knn.predict(X_entire)

print("")
print("Classification Report for whole dataset")
print("")
print(classification_report(y_entire, predict_y))


# Logistic Regression
print("")
print("")
print("##### Logistic Regression")

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
modelpredict = logreg.predict(X_test)
y_logreg_predict_entire = logreg.predict(X_entire)

print("")
print("Classification Report for test records")
print("")
print(classification_report(y_test, modelpredict))

print("")
print("Classification Report for whole dataset")
print("")
print(classification_report(y_entire, y_logreg_predict_entire))


# Multi-layer Perceptron
print("")
print("")
print("##### Multi-layer Perceptron")

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (3,  2), random_state = 1)
clf.fit(X_train, y_train)
y_clf_predict = clf.predict(X_test)

print("")
print("Classification Report for test records")
print("")
print(classification_report(y_test, y_clf_predict))

print("")
print("Classification Report for whole dataset")
print("")
y_clf_predict_entire = clf.predict(X_entire)

print(classification_report(y_entire, y_clf_predict_entire))


# Gaussian Naive Bayes
print("")
print("")
print("##### Gaussian Naive Bayes")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_gnb_predict = gnb.predict(X_test)

print("")
print("Classification Report for test records")
print("")
print(classification_report(y_test, y_gnb_predict))

print("")
print("Classification Report for whole dataset")
print("")
y_gnb_predict_entire = gnb.predict(X_entire)

print(classification_report(y_entire, y_gnb_predict_entire))

