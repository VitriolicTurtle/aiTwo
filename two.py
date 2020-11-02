import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

#   Iris
iris = datasets.load_iris()
irisX = iris.data
irisY = iris.target


#-------------------------------------------------------------------------------------------------------
#                                       Elbow method.
#                                       Iris Dataset.
#-------------------------------------------------------------------------------------------------------
#   Arrays where the return data is stored.
distortions = []
inertias = []
#   Relevant cluster range.
clusters = range(2,6)
for clu in clusters:
    #   Building and fitting the model
    irisKMeans = KMeans(n_clusters=clu).fit(irisX)
    #   The average of squared distance between centers of clusters.
    distortions.append(sum(np.min(cdist(irisX, irisKMeans.cluster_centers_, 'euclidean'),axis=1)) / irisX.shape[0])
    #   Distances between the closest cluster center and sample squared and summed.
    inertias.append(irisKMeans.inertia_)

#   Draw the graphs on same plot.
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Elbow method')
ax1.plot(clusters, distortions, 'bx-')
ax1.set_title('Distortion')

ax2.plot(clusters, inertias, 'bx-')
ax2.set_title('Inertia')
plt.show()

#-------------------------------------------------------------------------------------------------------
#                                       Silhouette Method.
#                                         Iris Dataset.
#-------------------------------------------------------------------------------------------------------
clusters = range(2,6)
for clu in clusters:                                    #   For each possible amount of cluster, run the KMeans.
    fig, silhouettes = plt.subplots(1, 1)
    irisKMeans = KMeans(n_clusters=clu)                 #   Run KMeans
    cluLabels = irisKMeans.fit_predict(irisX)           #   Runs clustering while returning labels (on Iris dataset.)
    silSamples = silhouette_samples(irisX, cluLabels)   #   Returns Silhouette coefficient (How well samples are clustered together with ones like themselves)
    yLow = 10                                           #   Determines bottom of cluster in graph
    for values in range(clu):                           #   Loop for each cluster in every version of the KMeans function:
        cluValues = silSamples[cluLabels == values]     #   Get sample where label (cluster nr) is the same as iterated value in clusternr array
        cluValues.sort()                                #   Otherwise, all data will look scrambeled.
        szCluValues = cluValues.shape[0]                #   Amount of values in each cluster (Height of cluster graph drawings)
        yHigh = yLow + szCluValues                      #   Determine the top value for the area that will be drawn in graph
        color = cm.nipy_spectral(float(values) / clu)   #   Determine colour
                                                        #   Fill in the cluster areas in graph
        silhouettes.fill_betweenx(np.arange(yLow, yHigh), 0, cluValues, facecolor=color, edgecolor="black", alpha=0.1)
                                                        #   Place the text on left side in center of the height of clusters
        silhouettes.text(-0.02, yLow + 0.5 * szCluValues, str(values))

        yLow = yHigh + 10                               #   Increase value to prepare for next iterations cluster placement.
    silAvg = silhouette_score(irisX, cluLabels)         #   Gets average value then draws a line through it below.
    silhouettes.axvline(x=silAvg, color="green", linestyle="-")
    fig.suptitle('Silhouette method')                   #   Displays title.

plt.show()
