import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from colorsys import hsv_to_rgb

plt.style.use("dark_background")

fr = 60  # framerate
time = 30  # amount of time

# num spirals
nspiral = 12.5
# frequencies
fspiral = 50
fpulse = 30
fcolor = 1
fpitch = 5
froll = 2

mpl.rcParams["toolbar"] = "none"

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
(line,) = ax.plot(np.nan, np.nan, np.nan)
ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(0, 1))
ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.canvas.toolbar_visible = False

s = np.linspace(0, 1, 10000)  # some arclength
tau = 2 * np.pi


def update(frame):
    t = frame / (fr * time)

    z = s
    x = np.sin(tau * (nspiral * s + t * fspiral)) * (
        1 + 0.2 * np.cos(s * 4 + fpulse * tau * t)
    )
    y = np.cos(tau * (nspiral * s + t * fspiral)) * (
        1 + 0.2 * np.cos(s * 4 + fpulse * tau * t + 2)
    )

    data = np.stack([x, y, z])
    line.set_data_3d(data)
    ax.view_init(elev=360 * t * fpitch, azim=0, roll=360 * t * froll)
    line.set_color(hsv_to_rgb(fcolor * t, 1, 1))
    return line


anim = animation.FuncAnimation(fig, update, fr * time, interval=1000 / fr)
plt.show()
