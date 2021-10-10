


import numpy as np
from dataCenter.dataCenter import dataSet
from kmodes.kprototypes import KPrototypes
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class kmeansModel():
    kproto = None
    def __init__(self, n_clusters):
        self.kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)

    kproto = KPrototypes(n_clusters=4, init='Cao', verbose=2)

    def Kprototype(self, varList):

        X = dataSet.vaccineData[varList + [dataSet.dependentVar]]
        X = X[X[dataSet.dependentVar] == "no_vacc"]
        X = X[varList]
        X = X.dropna(axis=0)
        X = X.sample(10000)

        # X["credit_hh_nonmtgcredit_60dpd"].quantile(0.95)


        a = X.dtypes
        float64var = a[a == "Float64"].index.to_list()
        categoricalvar = a[a == "object"].index.to_list()
        X[float64var] = (X[float64var] - X[float64var].mean()) / X[float64var].std()
        Xf = X[float64var]
        Xf = Xf.applymap(lambda x: min(x, 3) if x>0 else max(x, -3))

        X = X[categoricalvar + float64var]

        X = X[float64var]

        cateVarcolnum = [i for i in range(0, len(categoricalvar))]

        kproto = self.kproto

        clusters = kproto.fit_predict(X, categorical=cateVarcolnum)

        # Print cluster centroids of the trained model.
        print(kproto.cluster_centroids_)
        # Print training statistics
        print(kproto.cost_)
        print(kproto.n_iter_)

        kproto.labels_


        tsX = pd.get_dummies(X)
        transformer = SparsePCA(n_components=2, random_state=0)
        transformer.fit(tsX)
        X_transformed = transformer.transform(tsX)
        plt.scatter(X_transformed[:,0], X_transformed[:,1])
        plt.savefig("abc.png")
        plt.show()

        tsne = TSNE()

        tsne.fit_transform(tsX)

        pca = PCA(n_components=3)
        pca.fit(Xf)
        print(pca.explained_variance_ratio_)
        X_transformed = pca.transform(Xf)



        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xs = X_transformed[:, 0]
        ys = X_transformed[:, 1]
        zs = X_transformed[:, 2]
        ax.scatter(xs, ys, zs, marker="o")

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.savefig("abc.png")

        fig = plt.figure()
        xs = X_transformed[:, 0]
        ys = X_transformed[:, 1]
        plt.scatter(xs, ys, marker="o", s=0.5)

        plt.savefig("abcd.png")