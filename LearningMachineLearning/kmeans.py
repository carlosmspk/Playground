"""
    TO DO:
        -Preprocess: standard scaler

    Dataset:
        -Iris

    Methods:
    > Preprocessing:
        - PCA
        - ...
    > Clustering:
        - ...
"""

from sklearn import cluster
import utils
import os



if __name__ == '__main__':
    os.chdir("LearningMachineLearning")
    df_raw = utils.collect_data(visualize=0)
    df_preprocessed = utils.preprocess_data(df_raw, 'PCA', (2,))

    train_data = (df_preprocessed.drop(columns='Class').to_numpy())

    x = df_raw.iloc[:,[0,1,2,3]].values

    utils.elbow_kmeans(x)

    kmeans3 = cluster.KMeans(n_clusters=3)
    y_kmeans3 = kmeans3.fit_predict(x)
    print(kmeans3.cluster_centers_)

    import matplotlib.pyplot as plt
    plt.scatter(x[:,0], x[:,1], c=y_kmeans3, cmap='rainbow')
    plt.show()
