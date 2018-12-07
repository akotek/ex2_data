import numpy as np
import json
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from sklearn import datasets

# -------------------
# Code for ex2 in data science course:
# Part1 is data generation while Part2 is running a clustering algorithm:
# -------------------

# ----------------------------------------------------------------------
# Code from here belongs to part 1 of the eX:
# Generation of synthetic data of:
# Uniform dist, gaussian dist, circle in a ring dist and letters dist.
# ----------------------------------------------------------------------

# Constants
# -------------
AMOUNT = 300
# -------------


def gen_uniform_dist(low, high):
    x = np.random.uniform(low[0], low[1], AMOUNT)
    y = np.random.uniform(high[0], high[1], AMOUNT)
    return x, y


def gen_gaussian_dist(mu, sigma):
    pts_arr = np.random.multivariate_normal(mu, sigma, AMOUNT)  # [[x, y]] 2d np.array
    return pts_arr[:, 0], pts_arr[:, 1]


def gen_circle_inside_ring():
    x, y = datasets.make_circles(n_samples=AMOUNT, noise=.03, factor=.5)
    return x[:, 0], x[:, 1], y


def gen_first_letters():
    # Generates 2d points that represent
    # letters of AK (Artiyum and Kotek):
    return gen_A_char(), gen_K_char()


def gen_A_char():
    # TODO complete
    x, y = 1, 2
    return x, y


def gen_K_char():
    points = []
    # First line of K (x=100):
    points.append(np.ones(100) + gen_noise(100))
    points.append(np.arange(0, 100) + gen_noise(100))

    # Second and third lines: y=x, y=-x:
    points.append(np.arange(1, 51) + gen_noise(50))
    points.append(np.arange(51, 101) + gen_noise(50))
    points.append(np.arange(1, 51) + gen_noise(50))
    points.append(np.arange(1, 51)[::-1] + gen_noise(50))

    x, y = np.concatenate(points[::2]), np.concatenate(points[1::2])
    return x, y


def gen_noise(size):
    return 3 * np.random.rand(size)


def plot(x, y):
    plt.figure()
    plt.plot(x, y, 'o')
    plt.show()


def scatter(x, y, c=None):
    plt.figure()
    plt.scatter(x, y, c=c)
    plt.show()


def q3():
    # plot TWICE 300x:
    for i in range(2):
        pass
        # ------------------- 3.a
        # x1, y1 = gen_uniform_dist([-1, 1], [0, 5])
        # plot(x1, y1)
        # ------------------- 3.b
        # x2, y2 = gen_gaussian_dist(([1, 1]), np.array([[4, 0], [0, 4]]))  # COV is diagonal (4,0,0,4)
        # plot(x2, y2)
        # ------------------- 3.c
        # np.array of [ [x1],[y1], [x2],[y2], .... ]
        # we'll merge even/odd rows and get merged X and merged Y arrays and plot:
        # gauss_merged = np.concatenate([gen_gaussian_dist(([i, -i]), ([[2 * i, 0], [0, 2 * i]])) for i in range(1, 4)])
        # x3, y3 = np.concatenate(gauss_merged[::2]), np.concatenate(gauss_merged[1::2,:])
        # plot(x3, y3)
        # ------------------- 3.d
        # x4, y4, color = gen_circle_inside_ring()
        # plt.scatter(x4, y4, color)
        # ------------------- 3.e
        # a_tuple, k_tuple = gen_first_letters()
        # plot(a_tuple[0], a_tuple[1])
        # plot(k_tuple[0], k_tuple[1])


# ----------------------------------------------------------------------
# Code from here belongs to part 2 of the eX:
# -------------------
# Performing an hierarchical clustering of one-dim set of points
# from a json file,
# We'll be using pandas && json packages for parsing and analysing:
# -------------------
# ----------------------------------------------------------------------


# Constants
# -------------
DOLLAR = '$'


# -------------


def get_price_from_json(data):
    price = []
    for record in data['records']['record']:
        if record['Currency'] == DOLLAR:
            price.append(record['DolarsPelged'])
            # print("id is: %s , price is %d" % (record['id'], record['DolarsPelged']))
    return np.array(price)


def expand_dim(x):
    return np.column_stack((x, np.zeros(x.shape[0])))


def q4():
    with open('ex1.json', encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        price = get_price_from_json(data)
        price = expand_dim(price)  # algorithm works only on 2D space

        x = []
        y = []
        Z = linkage(price, 'ward')
        hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
        hc.fit_predict(price)

        i = 0
        while True:
            x.append(i)
            i += 1


q4()
