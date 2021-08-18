import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 0*np.outer(np.cos(u), np.sin(v))
y = 0*np.outer(np.sin(u), np.sin(v))
z = 0*np.outer(np.ones(np.size(u)), np.cos(v))

ln = ax.plot_surface(x, y, z, alpha=0.5)

def init():
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)
    return ln,


def update(frame):
    ax.clear()
    x = frame/10*np.outer(np.cos(u), np.sin(v))
    y = frame/10*np.outer(np.sin(u), np.sin(v))
    z = frame/10*np.outer(np.ones(np.size(u)), np.cos(v))
    ln = ax.plot_surface(x, y, z, alpha=0.5)
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)
    return ln,

from matplotlib.animation import FuncAnimation

anim = FuncAnimation (fig, func=update, frames=range(100), init_func=init, interval=2)

plt.show()