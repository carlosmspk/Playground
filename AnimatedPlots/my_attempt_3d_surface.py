# THIS CODE ISN'T WORKING YET

import numpy as np


def permutation_matrix_3d (res : int = 25, other_params : int = 0, param_seperated : bool = False):
    cols = 3 + other_params
    # We'll make all other parameters vary together from 0 to 1. This will originate the animation. If other_params = 1, this creates a full view of the universe, if other_params > 1 this will no longer be the case, and will only give a slightly more detailed view of the first 3 params as the other other_params change from 0 to 1 (in unison)
    if other_params > 0:
        rows = res**4
        perm_matrix = np.zeros((res,res,res,res,cols))
        perm_matrix[:,:,:,:,0] = np.linspace(0,1,res)
        perm_matrix[:,:,:,:,1] = np.linspace(np.zeros((res,)),np.ones((res,)),res)
        perm_matrix[:,:,:,:,2] = np.linspace(np.zeros((res,res)),np.ones((res,res)),res)
        perm_matrix[:,:,:,:,3:] = np.linspace(np.zeros((res,res,res,other_params)),np.ones((res,res,res,other_params)),res)
    else:
        rows = res**3
        perm_matrix = np.zeros((res,res,res,3))
        perm_matrix[:,:,:,0] = np.linspace(0,1,res)
        perm_matrix[:,:,:,1] = np.linspace(np.zeros((res,)),np.ones((res,)),res)
        perm_matrix[:,:,:,2] = np.linspace(np.zeros((res,res)),np.ones((res,res)),res)
    #this convoluted shape was only created in order to use numpy's speed, with linspace, we can now use a shape that is actually convenient for us:
    perm_matrix = np.reshape(perm_matrix, (rows, cols))
    return perm_matrix


def add_radius_flag (input : np.ndarray, center : np.ndarray = None):
    """
    For each point we want to do:
        (x-x_center)^2 + (y-y_center)^2 + (z-z_center)^2 <= r^2?
        if yes: append 'True' flag
        if no: append 'False' flag
    """
    if center is None:
        center = np.array((0.5,0.5,0.5))
    pass

    flags = ((input[:,0]-center[0])**2 + (input[:,1]-center[1])**2 + (input[:,2]-center[2])**2) <= input[:,3]
    flags = np.array(flags, dtype=int)
    flags[flags==0] = -1
    flags = flags.reshape((flags.shape[0],1))
    new_arr = np.concatenate((input,flags), axis=1)
    return new_arr


def animate_sphere (res : int = 10, fps:int = 10) -> None:
    to_plot = add_radius_flag(permutation_matrix_3d(res, 1))
    to_plot = np.split(to_plot, res)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ln, = ax.plot_surface([],[],[],alpha=0.5)
    
    up = np.array(range(res-1))
    down = np.array(range(res))
    down = down[:0:-1]
    frames = np.concatenate((up,down))

    frame_text = ax.text(-0.1, -0.1, -0.1, "")
    params_text = ax.text(0.2, -0.1, -0.1, "")

    def init_graph():
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
        return ln,

    def update(frame):
        frame_text.set_text ("Frame: " + str((frame+1)%(len(up)+2)) + "/" + str(len(up)+1))
        params_text.set_text ("Param Values: " + str((frame+1)%(len(up)+2)) + "/" + str(len(up)+1))
        this_frame = to_plot[frame]
        params_text.set_text ("Radius: " + "{0:.3f}".format(this_frame[0,-2]))
        inside_sphere = this_frame[this_frame[:,4] == 1]

        xdata = inside_sphere[:,0]
        ydata = inside_sphere[:,1]
        zdata = inside_sphere[:,2]
        ln.set_data_3d(xdata, ydata, zdata)
        return ln,

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig=fig,
                        func=update,
                        frames=frames, 
                        init_func=init_graph,
                        interval=max(int(1000/fps),1))
    plt.show()
    pass


if __name__ == '__main__':
    perm_matrix = animate_sphere(25, fps=10)
