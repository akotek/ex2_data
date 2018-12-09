import numpy as np
import json
from matplotlib import pyplot as plt
from math import pi, cos, sin
from random import random

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


def gen_uniform_dist(low, high, amount=AMOUNT):
    x = np.random.uniform(low[0], low[1], amount)
    y = np.random.uniform(high[0], high[1], amount)
    return x, y


def gen_gaussian_dist(mu, sigma):
    pts_arr = np.random.multivariate_normal(mu, sigma, AMOUNT)  # [[x, y]] 2d np.array
    return pts_arr[:, 0], pts_arr[:, 1]


def gen_first_letters():
    # Generates 2d points that represent
    # letters of AK (Abramovich and Kotek):
    return gen_A_char(), gen_K_char()


def gen_A_char():
    points = []
    # Generation of /
    points.append(np.arange(0, 25, 0.5) + gen_noise(50))
    points.append(np.arange(0, 100, 2) + gen_noise(50))

    # Generation of \
    points.append(np.arange(25, 50, 0.5) + gen_noise(50))
    points.append(np.arange(100, 0, -2) + gen_noise(50))

    # Generation of -
    points.append(np.random.uniform(13.5, 37.5, 50))
    points.append(np.random.uniform(48, 52, 50))

    # concatenate /-\ to get A
    x, y = np.concatenate(points[::2]), np.concatenate(points[1::2])
    return x, y


def gen_K_char():
    points = []
    # First line of K (x=100):
    points.append(np.ones(50) + gen_noise(50))
    points.append(np.arange(0, 100, 2) + gen_noise(50))

    # Second and third lines: y=x, y=-x:
    points.append(np.arange(1, 51) + gen_noise(50))
    points.append(np.arange(51, 101) + gen_noise(50))
    points.append(np.arange(1, 51) + gen_noise(50))
    points.append(np.arange(1, 51)[::-1] + gen_noise(50))

    x, y = np.concatenate(points[::2]), np.concatenate(points[1::2])
    return x, y


def gen_noise(size):
    return 3 * np.random.rand(size)


def plot(x, y, title=""):
    plt.figure()
    plt.plot(x, y, 'o')
    plt.title(title)
    plt.show()


def gen_ring(h, k, r1, r2, amount):
    """
    Creates points between r1 and r2 radiuses around the center of (h, k)
    :param h:
    :param k:
    :param r1:
    :param r2:
    :param amount:
    :return:
    """
    x = []
    y = []
    for i in range(amount):
        theta = random() * 2 * pi
        r = np.random.uniform(r1, r2)
        x.append(h + cos(theta) * r)
        y.append(k + sin(theta) * r)
    return x, y


def q3():
    """
    Makes the plots for question 3
    :return:
    """
    # plot TWICE 300x:
    for i in range(2):
        pass
        # ------------------- 3.a
        x1, y1 = gen_uniform_dist([-1, 1], [0, 5])
        plot(x1, y1, 'Question 3 (a) plot number %d' % (i + 1))
        # ------------------- 3.b
        x2, y2 = gen_gaussian_dist(([1, 1]), np.array([[4, 0], [0, 4]]))  # COV is diagonal (4,0,0,4)
        plot(x2, y2, 'Question 3 (b) plot number %d' % (i + 1))
        # ------------------- 3.c
        labels = []
        for j in range(1, 4):
            x, y = gen_gaussian_dist([j, -j], [[2 * j, 0], [0, 2 * j]])
            labels.append('(%d, %d) gaussian with std %d' % (j, -j, 2 * j))
            plt.plot(x, y, 'o')
        plt.legend(labels)
        plt.title('Question 3 (c) plot number %d' % (i + 1))
        plt.show()
        # ------------------- 3.d
        x4a, y4a = gen_ring(5, 5, 0, 2, 100)
        x4b, y4b = gen_ring(5, 5, 4, 5, 200)
        x4 = np.concatenate((x4a, x4b))
        y4 = np.concatenate((y4a, y4b))
        plot(x4, y4, 'Question 3 (d) plot number %d' % (i + 1))
        plt.show()
        # ------------------- 3.e
        a_tuple, k_tuple = gen_first_letters()
        plt.plot(a_tuple[0], a_tuple[1], 'o')
        plt.plot(k_tuple[0] + max(a_tuple[0] + 5), k_tuple[1], 'o')
        plt.title('Question 3 (e) plot number %d' % (i + 1))
        plt.show()


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
    """
    Get an array of prices from json
    :param data:
    :return:
    """
    price = []
    for record in data['records']['record']:
        if record['Currency'] == DOLLAR:
            price.append(record['DolarsPelged'])
    return np.array(price)


def find_closest_clusters(v, metric):
    """
    Finds 2 closest clusters in v (array of clusters) using metric callable
    :param v:
    :param metric:
    :return: tuple containing 1) the distance of merged clusters 2) the index of the first cluster that merged
                    3) the index of the second cluster that merged
    """
    distances = []
    for i in range(len(v)):
        for j in range(len(v)):
            if i != j:
                distances.append((metric(v[i], v[j]), i, j))
    return min(distances)


def perform_clustering(v, metric):
    """
    Clusters elements in array v using metric callable for distance evaluation
    :param v:
    :param metric:
    :return: the distances of merged cluster for every iteration
    """
    v = [[i] for i in v]
    distances_vector = []
    while len(v) > 1:
        min_distance, i, j = find_closest_clusters(v, metric)
        v[i] += v[j]
        del v[j]
        distances_vector.append(min_distance)
    return distances_vector


def num_of_clusters_at_threshold(threshold, distance_vector):
    """
    Finds how many clusters left just before hitting the closes cluster distance threshold
    :param threshold:
    :param distance_vector: distances of clusters that were merged by num of iteration
    :return:
    """
    for i in range(len(distance_vector)):
        if distance_vector[i] >= threshold:
            return len(distance_vector) - i + 1


def q4():
    """
    Makes the plots for question 4
    :return:
    """
    with open('ex1.json', encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        prices = get_price_from_json(data)
    distances_vector1 = perform_clustering(prices, lambda v1, v2: abs(np.average(v1) - np.average(v2)))
    distances_vector2 = perform_clustering(prices, lambda v1, v2: max(v1 + v2) - min(v1 + v2))
    plt.scatter(range(1, len(distances_vector1) + 1), distances_vector1)
    plt.title('Clustering by closest centroid distance.')
    plt.xlabel('Clustering number')
    plt.ylabel('Clustering distance')
    plt.show()
    plt.scatter(range(1, len(distances_vector2) + 1), distances_vector2)
    plt.title('Clustering by smallest diameter.')
    plt.xlabel('Clustering number')
    plt.ylabel('Clustering distance')
    plt.show()
    print('There are %d clusters before hitting the 0.5m threshold for minimal centroid distance clustering' %
          num_of_clusters_at_threshold(0.5 * distances_vector1[-1], distances_vector1))
    print('There are %d clusters before hitting the 0.1m threshold for minimal centroid distance clustering' %
          num_of_clusters_at_threshold(0.1 * distances_vector1[-1], distances_vector1))
    print('There are %d clusters before hitting the 0.5m threshold for minimal centroid diameter clustering' %
          num_of_clusters_at_threshold(0.5 * distances_vector2[-1], distances_vector2))
    print('There are %d clusters before hitting the 0.5m threshold for minimal centroid diameter clustering' %
          num_of_clusters_at_threshold(0.1 * distances_vector2[-1], distances_vector2))


q3()
q4()
