import json, pandas, os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

# -------------------
# Code for ex2 in data science course:
# Part1 is data generation while Part2 is running a clustering algorithm:
# -------------------


# Code from here belongs to part 1 of the eX:
# -------------------
# Generation of synthetic data of:
# Uniform dist, gaussian dist, circle in a ring dist and letters dist.
# -------------

# Constants
# -------------
AMOUNT = 100
# -------------

def gen_uniform_dist(low, high):
    x = np.random.uniform(low[0], low[1], AMOUNT)
    y = np.random.uniform(high[0], high[1], AMOUNT)
    return x, y


def gen_gaussian_dist(mu, sigma):
    pts_arr = np.random.multivariate_normal(mu, sigma, AMOUNT)  # [[x, y]] 2d np.array
    return pts_arr[:, 0], pts_arr[:, 1]


def gen_circle_inside_ring():
    x, y = datasets.make_circles(n_samples=AMOUNT, factor=0.4)
    return x[:, 0], x[:, 1]


def gen_first_letters():
    # Generates 2d points that represent
    # letters of AK (Artiyum and Kotek):
    #TODO complete
    pass


def plot(x, y):
    plt.figure()
    plt.plot(x, y, 'o')
    plt.show()


def q3():
    # ------------------- 3.a
    x1, y1 = gen_uniform_dist([-1, 1], [0, 5])
    # plot(x1, y1)
    # ------------------- 3.b
    x2, y2 = gen_gaussian_dist(([1, 1]), np.array([[4, 0], [0, 4]]))  # COV is diagonal (4,0,0,4)
    # plot(x2, y2)
    # ------------------- 3.c
    # np.array of [ [x1],[y1], [x2],[y2], .... ]
    # we'll merge even/odd rows and get merged X and merged Y arrays and plot:
    gauss_merged = np.concatenate([gen_gaussian_dist(([i, -i]), ([[2 * i, 0], [0, 2 * i]])) for i in range(1, 4)])
    x3, y3 = np.concatenate(gauss_merged[::2]), np.concatenate(gauss_merged[1::2,:])
    # plot(x3, y3)
    # ------------------- 3.d
    x4, y4 = gen_circle_inside_ring()
    # plot(x4, y4)
    # ------------------- 3.e
    x5, y5 = gen_first_letters()
    # plot (x5, y5)


# Code from here belongs to part 2 of the eX:
# -------------------
# Performing an hierarchical clustering of one-dim set of points
# from a json file,
# We'll be using pandas && json packages for parsing and analysing:
# -------------------

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def q4():
    data = json.loads(relpath("ex1.json"))


q3()
