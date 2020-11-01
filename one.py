import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA


#   Iris
iris = datasets.load_iris()
irisX = iris.data
irisY = iris.target

#   Fashion-mnist
mnistTrain = pd.read_csv(r"fashion-mnist_train.csv")
mnistTrainData = np.array(mnistTrain, dtype='float32')
xTrainFMN = mnistTrainData[:,1:] / 255
yTrainFMN = mnistTrainData[:,0]

#   STL10
with open("train_X.bin", 'rb') as f:
    everything = np.fromfile(f, dtype=np.uint8)
    xTrainSTL = np.reshape(everything, (-1, 3, 96, 96))
with open("train_y.bin", 'rb') as f:
    yTrainSTL = np.fromfile(f, dtype=np.uint8)

#-------------------------------------------------------------------------------------------------------
#   Plot the iris dataset clusters
irisKMeans = KMeans(n_clusters=3, random_state=0).fit(irisX)
irisLabel = [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]   #Use labels only on this one because its just 3 clusters
ax = Axes3D(plt.figure(1, figsize=(6,5)), elev=48, azim=134)
for name, label in irisLabel:
    ax.text3D(irisX[irisY == label, 3].mean(),
              irisX[irisY == label, 0].mean(),
              irisX[irisY == label, 2].mean()+2, name,
              horizontalalignment='center', bbox=dict(alpha=.0, edgecolor='w', facecolor='w'))
ax.scatter(irisX[:, 3], irisX[:, 0], irisX[:, 2], c=irisKMeans.labels_.astype(np.float), edgecolor='k')
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.dist = 12
plt.show()
#-------------------------------------------------------------------------------------------------------
#   Plot the Fashion-Mnist dataset clusters
#   Use PCA to get components describing position in 3-Dimensional space.
PCAdata = PCA(n_components=3).fit_transform(xTrainFMN)
fMnistKMeans = KMeans( n_clusters = 10, n_init = 1).fit(xTrainFMN, yTrainFMN)
ax = Axes3D(plt.figure(2, figsize=(8,7)), elev=48, azim=134)
ax.scatter(PCAdata[:, 2], PCAdata[:, 0], PCAdata[:,1], c=fMnistKMeans.labels_.astype(np.float), edgecolor='k')
plt.show()

#-------------------------------------------------------------------------------------------------------
#   Plot the STL10 dataset clusters
#   Reshapes 3D array into 2d
xTrainS2D, nx, ny, nz = xTrainSTL.shape
xTrainSTwo = xTrainSTL.reshape((xTrainS2D,nx*ny*nz))
#   Use PCA to get components describing position in 3-Dimensional space.
PCAdata = PCA(n_components=3).fit_transform(xTrainSTwo)
stlKMeans = KMeans( n_clusters = 10, n_init = 1).fit(xTrainSTwo, yTrainSTL)
ax = Axes3D(plt.figure(3, figsize=(8,7)), elev=48, azim=134)
ax.scatter(PCAdata[:, 2], PCAdata[:, 0], PCAdata[:,1], c=stlKMeans.labels_.astype(np.float), edgecolor='k')
plt.show()
#-------------------------------------------------------------------------------------------------------
