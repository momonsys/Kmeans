import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#x = np.arange(1,1001,1)
coor,label = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)
#dataset = list(zip(x,y))
n = np.arange(1,11,1)
inertial = []

for i in n:
      kmeans = KMeans(n_clusters=i)
      kmeans.fit(coor)
      inertial.append(kmeans.inertia_)

model=KMeans(n_clusters=3)
model.fit(coor)
#print(x)
#visualisasi
fig,axs = plt.subplots(1,2, figsize=(10,5))

axs[0].scatter(coor[:,0],coor[:,1], c = model.labels_, label= 'Data 1')
axs[0].set_title("Grafik Kmeans ")
axs[0].legend()

axs[1].scatter(n, inertial, marker='o',label='Data 2')
axs[1].set_title("Grafik Inertial ")
axs[1].legend()
plt.show()

