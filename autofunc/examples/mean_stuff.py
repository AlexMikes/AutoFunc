from statistics import mean
from statistics import harmonic_mean as hm
from scipy.stats.mstats import gmean as gm
from numpy import linspace as ls
import numpy as np
import matplotlib.pyplot as plt


# xs = ls(0,1,100)
# ys = ls(0,1,100)


xs = np.arange(0.0, 1.0, 0.1)
ys = np.arange(0.0, 1.0, 0.1)

fig, ax = plt.subplots()
ax.plot(xs,ys)

h = []

for x in xs:
    h.append((x,hm((x,x))))


# ax.plot(x, mean(x,y))

ax.grid(True, linestyle='-.')
ax.tick_params(labelcolor='r', labelsize='medium', width=3)

plt.show()



