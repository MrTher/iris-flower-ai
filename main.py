import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# printing information based on the data

iris_dataset = load_iris()
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("Target: \n{}".format(iris_dataset['target']))

x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X train shape: {}".format(x_train.shape))
print("Y train shape: {}".format(y_train.shape))

# using the k-nearest neighbour algorithm

knn = KNeighborsClassifier(n_neighbors=1)

# building the training model

knn.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new shape: {}".format(x_new.shape))

# getting the prediction by the ai

prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Prediction target name: {}".format(iris_dataset['target_names'][prediction]))

# checking if the result given by the ai is accurate enough lol

actual_result = knn.predict(x_test)
print("Actual target: {}".format(actual_result))
accuracy = str(round(np.mean(actual_result == y_test) * 100)) + "%"
print("AI Accuracy: {}".format(accuracy))
