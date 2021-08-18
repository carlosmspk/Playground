from numpy.lib.arraysetops import isin
import utils
from sklearn.svm import OneClassSVM
import numpy as np
import os

os.chdir("LearningMachineLearning")

def __visualize_svm(predictor, resolution : int):
    search_space = utils.permutation_matrix(res=resolution, simplified_features=2)

    results = predictor.predict(search_space)
            
    import matplotlib.pyplot as plt

    plt.title('SVM mapping for features 3 and 4 = 0')
    plt.scatter(search_space[:,0], search_space[:,1], c=results)
    plt.show()

def animate_svm(predictor, res:int, fps:int = 100):

    search_space = utils.permutation_matrix(res=res, simplified_features=2)
    results = predictor.predict(search_space)
    results = np.expand_dims(results, 1)
    results = np.concatenate((search_space,results), axis=1)
    to_plot = np.split(results, len(results)/(res*res))

    up = np.array(range(res-1))
    down = np.array(range(res))
    down = down[:0:-1]
    frames = np.concatenate((up,down))

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ln, = plt.plot([], [], 'o')
    frame_text = ax.text(-0.1, -0.1, "")
    params_text = ax.text(0.2, -0.1, "")

    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ln,

    def update(frame):
        frame_text.set_text ("Frame: " + str((frame+1)%(len(up)+2)) + "/" + str(len(up)+1))
        this_frame = to_plot[frame]
        params_text.set_text ("Param Values: " + "{0:.3f}".format(this_frame[0,-2]))
        outliers = this_frame[this_frame[:,4]==-1]
        xdata = outliers[:,0]
        ydata = outliers[:,1]
        ln.set_data(xdata, ydata)
        return ln, frame_text, params_text


    from matplotlib.animation import FuncAnimation
    
    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=False, interval=max(int(1000/fps),1))
    plt.show()
        


if __name__ == '__main__':
    df_raw = utils.collect_data(visualize=0)
    df_normalized = utils.preprocess_data(df_raw, 'Normalize', ('MinMax',))

    #split data into normal data and anomalous data
    df_train, df_outlier = utils.outlier_normal_split (df_normalized, split=True)

    model = OneClassSVM(gamma='scale', nu=0.01)

    #fit SVM to normal data
    train_data = df_train.values
    model.fit(train_data)
    
    animate_svm(model, 25)