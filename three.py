import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.image as mpimg


imgData = mpimg.imread('threeImage.png')                #   Read image.
imgHeight, imgWidth, channels = imgData.shape           #   Fetch shape data (477, 779, 4).
                                                        #   reshape data by multiplying image height with number of channels.
reshapedData = imgData.reshape((imgHeight,imgWidth*channels))


clusterValues = [50, 25, 10]                            #   List of clusters used in compression.
compressedImages = list()                               #   Will hold data of images after compression.

for clu in clusterValues:
    pcaData = PCA(clu).fit(reshapedData)
    transformedPca = pcaData.transform(reshapedData)
    #   Exvlusively used for displaying
    compressedImg = pcaData.inverse_transform(transformedPca)
    compressedImg = np.reshape(compressedImg, ( imgHeight, imgWidth, channels))
    compressedImages.append(compressedImg)              #   Put compressed image in list

#   Code necessary for displaying the 4 figures
fig, ax = plt.subplots(2, 2, figsize = (8, 5))
fig.suptitle('PCA Compression')
ax[0,0].axis('off')
ax[0,0].imshow(imgData)
ax[0,0].set_title("Original")
ax[0,1].axis('off')
ax[0,1].imshow(compressedImages[0])
ax[0,1].set_title("50 Clusters")
ax[1,0].axis('off')
ax[1,0].imshow(compressedImages[1])
ax[1,0].set_title("25 Clusters")
ax[1,1].axis('off')
ax[1,1].imshow(compressedImages[2])
ax[1,1].set_title("10 Clusters")
plt.show()



##################################################################################
#
#   TEST PCA COMPRESSION BY SEPARATING THE RGB CHANNELS AND USING PCA ON EACH
#   CONCLUSION: No difference from above solution
#
##################################################################################

#   Very hardcoded way to separate the RGBA channels
pxR = np.delete(imgData, 3, 2)
pxR = np.delete(pxR, 2, 2)
pxR = np.delete(pxR, 1, 2)

pxG = np.delete(imgData, 3, 2)
pxG = np.delete(pxG, 2, 2)
pxG = np.delete(pxG, 0, 2)

pxB = np.delete(imgData, 3, 2)
pxB = np.delete(pxB, 1, 2)
pxB = np.delete(pxB, 0, 2)

#   Separate width, height, and channels.
rnsamples, rnx, rny = pxR.shape
reR = pxR.reshape((rnsamples,rnx*rny))
gnsamples, gnx, gny = pxG.shape
reG = pxG.reshape((gnsamples,gnx*gny))
bnsamples, bnx, bny = pxB.shape
reB = pxB.reshape((bnsamples,bnx*bny))
#   Run PCA
pcaR = PCA(5).fit(reR)
pcaG = PCA(5).fit(reG)
pcaB = PCA(5).fit(reB)
tr = pcaR.transform(reR)
tg = pcaG.transform(reG)
tb = pcaB.transform(reB)
cr = pcaR.inverse_transform(tr)
cg = pcaG.inverse_transform(tg)
cb = pcaB.inverse_transform(tb)
cor = np.reshape(cr, (rnsamples,rnx*rny))
cog = np.reshape(cg, (gnsamples,gnx*gny))
cob = np.reshape(cb, (bnsamples,bnx*bny))
#   Create final image data.
rgb = np.dstack((cor,cog,cob))
plt.imshow(rgb)
plt.show()
