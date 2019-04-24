from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 3], [8, 3], [2, 6], [5, 2], [0, 6], [2, 4], [7, 1], [3, 0], [1, 3], [4, 0], [5, 4], [0, 6], [3, 1]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","b.","y."]

for i in range(len(x)):
    print("coordinate:",x[i], "label:", labels[i])
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize = 15)


plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 3, zorder = 10)
plt.show()