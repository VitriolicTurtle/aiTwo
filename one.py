#   DISABLE UNNECESSARY WARNING
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
#----------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import os
import struct
import array as pyarray

#   Iris
iris = datasets.load_iris()
irisX = iris.data
irisY = iris.target
xTrainIris, xTestIris, yTrainIris, yTestIris = train_test_split(irisX, irisY, test_size=0.2)
#   Fashion-mnist
#   Training set
mnistTrain = pd.read_csv(r"fashion-mnist_train.csv")
mnistTrainData = np.array(mnistTrain, dtype='float32')
xTrainFMN = mnistTrainData[:,1:] / 255
yTrainFMN = mnistTrainData[:,0]
#   Testing set
mnistTest = pd.read_csv(r"fashion-mnist_test.csv")
mnistTestData = np.array(mnistTest, dtype='float32')
xTestFMN = mnistTestData[:,1:] / 255
yTestFMN = mnistTestData[:,0]

#   STL10
#   Training set
with open("train_X.bin", 'rb') as f:
    everything = np.fromfile(f, dtype=np.uint8) / 255
    xTrainSTL = np.reshape(everything, (-1, 3, 96, 96))
with open("train_y.bin", 'rb') as f:
    yTrainSTL = np.fromfile(f, dtype=np.uint8)
#   Testing set
with open("test_X.bin", 'rb') as f:
    everything = np.fromfile(f, dtype=np.uint8) / 255
    xTestSTL = np.reshape(everything, (-1, 3, 96, 96))
with open("test_y.bin", 'rb') as f:
    yTestSTL = np.fromfile(f, dtype=np.uint8)

#-------------------------------------------------------------------------------------------------------
#   Plot the iris dataset clusters
irisKMeans = KMeans(n_clusters=3, random_state=10).fit(xTrainIris)
irisLabel = [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]   #Use labels only on this one because its just 3 clusters
irisKMeansPred = irisKMeans.predict(xTestIris)
ax = Axes3D(plt.figure(1, figsize=(6,5)), elev=48, azim=134)
for name, label in irisLabel:
    ax.text3D(xTrainIris[yTrainIris == label, 3].mean(),
              xTrainIris[yTrainIris == label, 0].mean(),
              xTrainIris[yTrainIris == label, 2].mean()+2, name,
              horizontalalignment='center', bbox=dict(alpha=.0, edgecolor='w', facecolor='w'))
ax.scatter(xTrainIris[:, 3], xTrainIris[:, 0], xTrainIris[:, 2], c=irisKMeans.labels_.astype(np.float), edgecolor='k')
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.dist = 12
plt.show()
#-------------------------------------------------------------------------------------------------------
#   Plot the Fashion-Mnist dataset clusters
#   Use PCA to get components describing position in 3-Dimensional space.
PCAdata = PCA(n_components=3).fit_transform(xTrainFMN)
fMnistKMeans = KMeans( n_clusters = 10, n_init = 10).fit(xTrainFMN, yTrainFMN)
fMnistKMeansPred = fMnistKMeans.predict(xTestFMN)
ax = Axes3D(plt.figure(2, figsize=(8,7)), elev=48, azim=134)
ax.scatter(PCAdata[:, 2], PCAdata[:, 0], PCAdata[:,1], c=fMnistKMeans.labels_.astype(np.float), edgecolor='k')
ax.dist = 12
plt.show()
#-------------------------------------------------------------------------------------------------------
#   Plot the STL10 dataset clusters
#   Reshapes 3D array into 2d
xTrainS2D, nx, ny, nz = xTrainSTL.shape
xTrainSTwo = xTrainSTL.reshape((xTrainS2D,nx*ny*nz))
xTestS2D, nx, ny, nz = xTestSTL.shape
xTestS2DTwo = xTestSTL.reshape((xTestS2D,nx*ny*nz))
#   Use PCA to get components describing position in 3-Dimensional space.
PCAdata = PCA(n_components=3).fit_transform(xTrainSTwo)
stlKMeans = KMeans( n_clusters = 10, n_init = 10).fit(xTrainSTwo, yTrainSTL)
stlKMeansPred = stlKMeans.predict(xTestS2DTwo)
ax = Axes3D(plt.figure(3, figsize=(8,7)), elev=48, azim=134)
ax.scatter(PCAdata[:, 2], PCAdata[:, 0], PCAdata[:,1], c=stlKMeans.labels_.astype(np.float), edgecolor='k')
ax.dist = 12
plt.show()
#-------------------------------------------------------------------------------------------------------


print("")
print("")
print("                     IRIS DATASET:")
print(classification_report(yTestIris, irisKMeansPred))
print("")
print("")
print("                     FASHION-MNIST DATASET:")
print(classification_report(yTestFMN, fMnistKMeansPred))
print("")
print("")
print("                     STL10 DATASET:")
print(classification_report(yTestSTL, stlKMeansPred))
