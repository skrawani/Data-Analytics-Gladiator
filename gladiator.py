#--------------------First Phase---------------------

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iris.csv')

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

#PCA
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3)
X_sklearn = sklearn_pca.fit_transform(X)

#--------------------Second Phase---------------------

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#plotting the dataset
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
k = X_sklearn[:,0]
l = X_sklearn[:,1]
m = X_sklearn[:,2]

ax.scatter(k, l, m, c=y, marker='o')

ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.set_zlabel('Third Component')

plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y, test_size = 1/3, random_state = 20)

#--------------------Third Phase---------------------

# Euclidean distance
def distance(instance1, instance2):
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1) 
    instance2 = np.array(instance2)
    
    return np.linalg.norm(instance1 - instance2)


# Get Neighbours
def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  k, 
                  distance=distance):
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors  

# Counter
from collections import Counter
def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]

#Prediction for k = 1
y_pred1 = []
for i in range(50):
    neighbors = get_neighbors(X_train, y_train, X_test[i], 1, distance=distance)
    y_pred1.append(vote(neighbors))
y_pred1 = np.array(y_pred1)

#Prediction for k = 3
y_pred3 = []
for i in range(50):
    neighbors = get_neighbors(X_train, y_train, X_test[i], 3, distance=distance)
    y_pred3.append(vote(neighbors))
y_pred3 = np.array(y_pred3)

#Prediction for k = 5
y_pred5 = []
for i in range(50):
    neighbors = get_neighbors(X_train, y_train, X_test[i], 5, distance=distance)
    y_pred5.append(vote(neighbors))
y_pred5 = np.array(y_pred5)