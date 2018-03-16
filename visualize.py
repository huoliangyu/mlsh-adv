# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# data = np.random.rand(10, 10) * 20
# 0: left: red
# 1: down: blue
# 2: right: pink
# 3: up: cyan
l = 0
d = 1
r = 2
u = 3
# boundary
b = -1
# goal
g = -2

        # "XXGXXXXXX",
        # "XOOOXOOOX",
        # "XOOOOOOOX",
        # "XOOOXOOOX",
        # "XXOXXXOXX",
        # "XOOOXOOOX",
        # "XOOOOOOOX",
        # "XOOOXOOOX",
        # "XXXXXXXXX",
data = [
    [b, b, g, b, b, b, b, b, b],
    [b, r, u, l, b, d, d, d, b],
    [b, r, u, l, l, l, l, l, b],
    [b, r, u, l, b, u, u, u, b],
    [b, b, u, b, b, b, u, b, b],
    [b, r, u, l, b, d, d, d, b],
    [b, r, u, l, l, l, l, l, b],
    [b, r, u, l, b, u, u, u, b],
    [b, b, b, b, b, b, b, b, b],
]

# create discrete colormap
cmap = colors.ListedColormap(['green', 'black', 'red', 'blue', 'pink', 'cyan'])
bounds = [-2,-1,0,1,2,3,4]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(-.5, 9, 1));
ax.set_yticks(np.arange(-.5, 9, 1));
# ax.set_xticks([]);
# ax.set_yticks([]);
# plt.axis('off')
# https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

print 'data ='
print data

plt.show()