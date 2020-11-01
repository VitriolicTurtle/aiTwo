import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA

np.random.seed(5)

#   Iris
iris = datasets.load_iris()
irisX = iris.data
irisY = iris.target

#   Fashion-mnist
mnistTrain = pd.read_csv(r"data/fashion-mnist_test.csv")
#mnistTest = pd.read_csv(r"data/fashion-mnist_train.csv")
mnistTrainData = np.array(mnistTrain, dtype='float32')
#mnistTestData = np.array(mnistTest, dtype='float32')
x_trainMnist = mnistTrainData[:,1:] / 255
y_trainMnist = mnistTrainData[:,0]
#x_testMnist = mnistTestData[:, 1:] / 255
#y_testMnist = mnistTestData[:,0]
xTrainM, xValM, yTrainM, yValM = train_test_split(x_trainMnist, y_trainMnist, test_size=0.2)

with open("data/stl10_binary/train_X.bin", 'rb') as f:
    everything = np.fromfile(f, dtype=np.uint8)
    xTrainSTL = np.reshape(everything, (-1, 3, 96, 96))
    xTrainSTL = np.transpose(xTrainSTL, (0, 3, 2, 1))

with open("data/stl10_binary/train_y.bin", 'rb') as f:
    yTrainSTL = np.fromfile(f, dtype=np.uint8)

xTrainS, xValS, yTrainS, yValS = train_test_split(xTrainSTL, yTrainSTL, test_size=0.2)


print(xTrainSTL)
#---------------------------------------------------------------------------------------------------
irisKMeans = KMeans(n_clusters=3, random_state=0).fit(irisX)
irisLabel = [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]
fignum = 1
fig = plt.figure(fignum, figsize=(6,5))
ax = Axes3D(fig, elev=48, azim=134)
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
#plt.show()
#-------------------------------------------------------------------------------------------------------


PCAdata = PCA(n_components=3).fit_transform(xTrainM)
fMnistKMeans = KMeans( n_clusters = 10, n_init = 1).fit(xTrainM, yTrainM)
fig = plt.figure(fignum, figsize=(8,7))
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(PCAdata[:, 2], PCAdata[:, 0], PCAdata[:,1], c=fMnistKMeans.labels_.astype(np.float), edgecolor='k')

#plt.show()

#-------------------------------------------------------------------------------------------------------

PCAdata = PCA(n_components=3).fit_transform(xTrainS)
stlKMeans = KMeans( n_clusters = 10, n_init = 1).fit(xTrainS, yTrainS)
fig = plt.figure(fignum, figsize=(8,7))
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(PCAdata[:, 2], PCAdata[:, 0], PCAdata[:,1], c=stlKMeans.labels_.astype(np.float), edgecolor='k')

plt.show()
