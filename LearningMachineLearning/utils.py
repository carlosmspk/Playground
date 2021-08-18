from math import dist
from typing import Iterable
from numpy.lib.utils import info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn import cluster

def inf(*args, **kwargs):
    print ('[Info]', end=' ')
    print(*args, **kwargs)

    
def  war(*args, **kwargs):
    print ('[WARNING]', end=' ')
    print(*args, **kwargs)

    
def  err(*args, **kwargs):
    print ('![ERROR]!', end=' ')
    print(*args, **kwargs)


def create_color_class_dict (df:pd.DataFrame) -> dict:
    """convenience function to create color codings for different class names"""
    class_set = set() # {}
    color_set = {'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'lime', 'purple', 'orange'}

    for _, item in df.iterrows():
        class_set.add(item['Class'])

    class_to_color_dict = dict()
    for class_name in class_set:
        class_to_color_dict[class_name] = color_set.pop()

    return class_to_color_dict


def visualize_data (df:pd.DataFrame, rows:int = None, cols:int = None, title : str = None) -> None:
    """convenience function to plot data"""

    labels = list() # []
    for label in df:
        if label == 'Class':
            continue
        labels.append(label)
    
    plot_number = 0
    for label1, label2 in combinations(labels, 2):
        plot_number += 1

    class_to_color_dict = create_color_class_dict(df)
    
    nrows = rows if rows is not None else plot_number
    ncols = cols if cols is not None and rows is not None else 1

    #single plot
    if nrows == ncols == 1:
        added_legend_classes = set()
        for x, y, flower_class in zip(df[labels[0]],df[labels[1]], df['Class']):
            if flower_class in added_legend_classes:
                plt.scatter(x, y, color = class_to_color_dict[flower_class])
            else:
                added_legend_classes.add(flower_class)
                plt.scatter(x, y, color = class_to_color_dict[flower_class], label=flower_class)
            plt.legend()
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            if title:
                plt.title(title, fontsize=20)
    #multi plot
    else:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        for i, (axis, label_combo) in enumerate(zip(axs.flatten(), combinations(labels, 2)), start=1):
            added_legend_classes = set()
            label1, label2 = label_combo
            for x, y, flower_class in zip(df[label1],df[label2], df['Class']):
                if flower_class in added_legend_classes:
                    axis.scatter(x, y, color = class_to_color_dict[flower_class])
                else:
                    added_legend_classes.add(flower_class)
                    axis.scatter(x, y, color = class_to_color_dict[flower_class], label=flower_class)
            axis.set_xlabel(label1)
            axis.set_ylabel(label2)

            if i == ncols*nrows:
                handles, labels = axis.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        if title:
            fig.suptitle(title, fontsize=20)
            
    plt.show()


def collect_data(visualize:int = 2) -> pd.DataFrame:
    """
    visualize = 0 : No prints or graphs
    visualize = 1 : Prints dataframe head
    visualize = 2 : shows graph with all data-color combos coded by classification
    """
    #load data
    df = pd.read_csv('datasets\iris.csv')

    if visualize > 0:
        #check if data makes sense
        print(df.head())

    if visualize > 1:
        #plot data
        visualize_data(df, class_label='Class', rows=3, cols=2)
    return df


def print_k_means_inf(kmeans : cluster.KMeans) -> None:

    # The lowest SSE value
    print('Inertia:',kmeans.inertia_)
    # Final locations of the centroid
    print('Centers:',kmeans.cluster_centers_)
    # The number of iterations required to converge
    print('Iters:',kmeans.n_iter_)
    # first five predicted labels 
    print('Labels:',kmeans.labels_[:5])


def elbow_kmeans (input_data, k_search_space : int = 11, plot_results = True) -> tuple:
    error = []
    search_space = range(1, k_search_space)

    for i in search_space:
        kmeans = cluster.KMeans(n_clusters=i).fit(input_data)
        error.append(kmeans.inertia_)
    if plot_results:
        plt.scatter(search_space, error)
        plt.title("Elbow method's k vs error")
        plt.xlabel('Cluster number')
        plt.ylabel('Error')
        plt.show()

    return tuple(search_space, error)


def get_least_frequent(target : pd.DataFrame, class_label : str) -> str:
    """
    Returns the single most frequent value from series (string or int).
    """
    occurrences = pd.DataFrame(target[class_label].value_counts())
    outlier_class_label = occurrences.idxmin()[0]
    occurrences = occurrences.min()[0]
    inf("No outlier class defined. Defining '" + outlier_class_label + "' as outlier class with " + str(occurrences) + " occurrences.")
    return outlier_class_label


def outlier_normal_split(df : pd.DataFrame, outlier_class_label : str = None, class_label = 'Class', split = False):
    """
        Description
        ---
        Returns new dataframe with the class label split into '0' for normal and '1' for outlier. If split is True, returns two dataframes, one for each type of data in the format of (df_normals, df_outliers) and class labels are removed

        Parameters
        ---
         - df : pandas.Dataframe 
            dataframe to split.
         - outlier_class_label : str
            label of class to be considered anomalous, all other classes will be considered as normal. If None (default value) the single class with the highest occurrence will be considered normal.
         - class_label : Any
            the dataframe column name that holds the classification of each data points. Defaults to 'Class'.
         - split : bool
            whether to split the Dataframe into two Dataframes or not
    """

    if outlier_class_label is None:
        outlier_class_label = get_least_frequent(df, class_label=class_label)
    
    df_new = df.copy(deep=False)
    df_new.loc[df_new[class_label] == outlier_class_label, class_label] = 1
    df_new.loc[df_new[class_label] != 1, class_label] = 0

    if split:
        df_normal = df_new[df_new[class_label] == 0]
        df_outlier = df_new[df_new[class_label] == 1]
        df_normal = df_normal.drop(columns=class_label)
        df_outlier = df_outlier.drop(columns=class_label)
        return df_normal, df_outlier
    return df_new


def preprocess_data(df : pd.DataFrame, preprocess_method : str, args : tuple = (None,)) -> pd.DataFrame:

    ### PCA
    if preprocess_method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(args[0] or 3)
        pca_transformed_data = pca.fit_transform(df.drop(columns='Class'))
        new_df_dict = dict()
        for i, pca_component in enumerate(pca_transformed_data.transpose(), start=1):
            new_df_dict['PCA'+str(i)] = pca_component
        return pd.DataFrame(new_df_dict).join(other=df[['Class']])
    
    ### Use a normalizing scaler
    elif preprocess_method == 'Normalize':
        if args[0] is None or str(args[0]) == 'Standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif str(args[0]) == 'MinMax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError("Input first argument " + str(args[0]) + " doesn't match any scaler. Allowed scalers: 'Standard' or 'MinMax'")
        x = df.drop(columns='Class').values
        scaled_x = scaler.fit_transform(x)
        class_array = np.array(df['Class'])[:, np.newaxis]
        new_values = np.concatenate((scaled_x, class_array), axis = 1)
        scaled_df =  pd.DataFrame(new_values, columns=df.columns)
        return scaled_df
        

    else:
        raise NotImplementedError ("Preprocess method '" + preprocess_method + "' is not implemented. Available methods: 'PCA' or 'Normalize'")


def permutation_matrix (res:int = 25, simplified_features:int = 0, return_as_list:bool = False):
    simp_start = np.zeros((simplified_features,))
    simp_stop = np.ones((simplified_features,))
    rows = res*res*res
    cols = simplified_features + 2
    perm_matrix = np.zeros((rows,cols))

    for z in range(res):
        j = z*res*res
        for i in range(res):
            perm_matrix[j+i:j+res*res:res,1] = np.linspace(0,1,res)
            perm_matrix[j+i*res:j+i*res+res,0] = np.linspace(0,1,res)
            perm_matrix[z*res+i::res*res,2:cols] = np.linspace(simp_start,simp_stop, res)
    
    if return_as_list:
        samples = []
        for each_rep in range(res):
            samples.append(perm_matrix[each_rep*res*res:each_rep*res*res+res*res])
        print (samples)
        return samples
    else:
        return perm_matrix

