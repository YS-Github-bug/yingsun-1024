import numpy
import pandas
from sklearn import datasets
import scipy.cluster.hierarchy as hcluster
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

#pal = sns.color_palette("Paired", 10)
#pal = [mpl.colors.rgb2hex(x) for x in pal]
#hcluster.set_link_color_palette( pal )
hcluster.set_link_color_palette(['#a6cee3', '#1f78b4', '#b2df8a'])
iris = datasets.load_iris()
data = iris.data
target = iris.target
points=scipy.randn(20,4)
# Compute and plot first dendrogram.
linkage = hcluster.linkage(points, method='centroid')
P = hcluster.dendrogram(linkage,  leaf_font_size=10.)
#p = hcluster.fcluster( linkage,  3, criterion='maxclust')
for i, j in zip(P['color_list'], P['leaves']):
	print(i, j, sep  = '\t')
plt.show()
