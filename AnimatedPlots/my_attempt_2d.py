import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def permutation_matrix (res:int = 25, simplified_features:int = 0, return_as_list:bool = True):
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
        return samples
    else:
        return perm_matrix



fig, ax = plt.subplots()
to_plot = permutation_matrix(simplified_features=2)
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ln,

def update(frame):
    xdata = to_plot[frame][:,0]*25/frame
    ydata = to_plot[frame][:,1]*25/frame
    ln.set_data(xdata, ydata)
    return ln,

up = np.array(range(24))
down = np.array(range(25))
down = down[::-1]
frames = np.concatenate((up,down))


ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=100)
plt.show()