import numpy as np
from os.path import join
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Inputs for TSNE: provide distance matrix between all pairs of conformations. The RMSD.dat that I am using in the next line is the pairwise RMSD obtained from Gromacs in binary vector format.

data = np.fromfile('rmsd.dat', np.float32)
rmsd = data.reshape(int(np.sqrt(len(data))), int(np.sqrt(len(data)))) ### To reshape the vector to matrix

##To make sure if the matrix is symmetric
import sklearn.utils.validation as suv
suv.check_symmetric(rmsd, raise_exception=True)

with open('status.txt', 'a') as f1:
     f1.write("\n")
     print('symmetry check completed', file=f1)

# Creating the TSNE object and projection
perplexityVals = range(100, 2100, 100)
for i in perplexityVals:
    tsneObject = TSNE(n_components=2, perplexity=i, early_exaggeration=10.0, learning_rate=100.0, n_iter=3500, n_iter_without_progress=300, min_grad_norm=1e-7, metric="precomputed", init='random', method='barnes_hut', angle=0.5) ### metric is precomputed RMSD distance. if you provide Raw coordinates, the TSNE will compute the distance by default with Euclidean metrics
    tsne = tsneObject.fit_transform(rmsd)
    np.savetxt("tsnep{0}".format(i), tsne)

#Kmeans clustering
range_n_clusters = [20, 30, 40, 50, 60, 70, 80, 90, 100]
#range_n_clusters = [2, 4, 6, 8, 10, 12, 14, 16, 18]
perplexityVals = range(100, 2100, 100)
for perp in perplexityVals:
    tsne = np.loadtxt('tsnep'+str(perp))
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters).fit(tsne)
        np.savetxt('kmeans_'+str(n_clusters)+'clusters_centers_tsnep'+str(perp), kmeans.cluster_centers_, fmt='%1.3f')
        np.savetxt('kmeans_'+str(n_clusters)+'clusters_tsnep'+str(perp)+'.dat', kmeans.labels_, fmt='%1.1d')
#### Compute silhouette score based on low-dim and high-dim distances        
        silhouette_ld = silhouette_score(tsne, kmeans.labels_)
        silhouette_hd = metrics.silhouette_score(rmsd, kmeans.labels_, metric='precomputed')
        with open('silhouette.txt', 'a') as f:
            f.write("\n")
            print(perp, n_clusters, silhouette_ld, silhouette_hd, silhouette_ld*silhouette_hd, file =f)

##### plotting for the best cluster with highest silhouette score######       
s = np.loadtxt('silhouette.txt')
[bestP,bestK] = s[np.argmax(s[:,4]), 0], s[np.argmax(s[:,4]), 1]
besttsne = np.loadtxt('tsnep'+str(int(bestP)))
bestclust = np.loadtxt('kmeans_'+str(int(bestK))+'clusters_tsnep'+str(int(bestP))+'.dat')
plt.rc('font', family='sans-serif', weight='normal', size='14')
plt.rc('axes', linewidth=1.5)
cmap = cm.get_cmap('jet', bestK) 
plt.scatter(besttsne[:,0], besttsne[:,1], c= bestclust.astype(float), s=50, alpha=0.5, cmap=cmap)
plt.savefig('tsnep'+str(int(bestP))+'_kmeans'+str(int(bestK))+'.png', dpi=600)

##### Uncomment the next three lines for writing out all cluster members ####### 
#c_members = {i: np.where(bestclust == i)[0] for i in range(bestK)}
#with open('c_members'+str(bestK)+'.txt', 'w') as f:
#     print(c_members, file=f)

##### Uncomment the next five lines  for finding closest member of each cluster from its center point #######
#center = np.loadtxt('kmeans_'+str(int(bestK))+'clusters_centers_tsnep'+str(int(bestP))) #kmeans.cluster_centers_
#from scipy import spatial
#distance,index = spatial.KDTree(besttsne, leafsize=10).query(center, 10)
#with open('nearest10_cluster'+str(bestK)+'.txt', 'w') as f:
#     print(index+1, file=f)

####For clarification, Please contact anand@iisc.ac.in or rajeswari.biotech@gmail.com
