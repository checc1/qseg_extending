import numpy as np
import networkx as nx
from mnist_3D import xtrain
import pandas as pd


def gaussian_filter(a: np.ndarray, b: np.ndarray, sigma: float = 0.05) -> float:
    """
    Compute the Gaussian similarity between two neighbouring pixels.
    :param a: (np.ndarray) of shape representing the pixel value of the first pixel;
    :param b: (np.ndarray) of shape representing the pixel value of the second pixel.
    :param sigma : (float) the standard deviation of the Gaussian kernel.
    :return: np.ndarray of shape representing the Gaussian similarity between the two pixels.
    """
    assert 0 <= sigma <= 1, f"Sigma should be in [0, 1], got {sigma}"
    assert 0 <= a <= 1, f"Pixel value should be in [0, 1], got {a}"
    assert 0 <= b <= 1, f"Pixel value should be in [0, 1], got {b}"
    return np.exp(-((a - b) ** 2) / (2 * sigma ** 2))


def image_to_grid_graph(gray_img: np.ndarray) -> list[tuple]:
    """
    Convert a grayscale image to a grid graph with Gaussian similarity as edge weights.
    :param gray_img: (np.ndarray) of shape representing the grayscale image;
    :return: list of edges with weights for the graph.
    """
    h, w = gray_img.shape
    nodes = np.zeros((h * w, 1))
    edges = []
    nx_elist = []
    min_weight = 1
    max_weight = 0

    for i in range(nodes.shape[0]):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y]

        if x > 0 or y > 0:
              j = (x - 1) * w + y
              weight = 1 - gaussian_filter(gray_img[x, y], gray_img[x - 1, y])
              edges.append((i, j, weight))
              nx_elist.append(((x, y), (x - 1, y), np.round(weight, 4)))
              min_weight = min(min_weight, weight)
              max_weight = max(max_weight, weight)

        if y > 0:
              j = x * w + y - 1
              weight = 1 - gaussian_filter(gray_img[x, y], gray_img[x, y - 1])
              edges.append((i, j, weight))
              nx_elist.append(((x, y), (x, y - 1), weight))
              min_weight = min(min_weight, weight)
              max_weight = max(max_weight, weight)
    a = -1
    b = 1
    if max_weight - min_weight:
        normalized_nx_elist = [
            (node1, node2, -1 * np.round(((b - a) * ((edge_weight - min_weight) / (max_weight - min_weight))) + a, 4))
            for node1, node2, edge_weight in nx_elist]
    elif max_weight == 0 and min_weight == 0:
        normalized_nx_elist = [(node1, node2, 1) for node1, node2, edge_weight in nx_elist]
    else:
        normalized_nx_elist = [(node1, node2, -1 * (np.round(edge_weight, 4))) for node1, node2, edge_weight in nx_elist]
    return normalized_nx_elist


def print_info(graph: nx.Graph) -> None:
    """
    Print the information about the graph G.
    :param graph: (nx.Graph) the graph to print information about.
    :return: df (pd.DataFrame) the dataframe containing the information.
    """
    edge_labels = list(nx.get_edge_attributes(graph, "weight").values())
    data = {"Max": float(np.max(edge_labels)),
               "Min": float(np.min(edge_labels)),
               "Mean": float(np.mean(edge_labels))}

    df = pd.DataFrame(data, index=[0])
    print(df)


loaded_img = xtrain[0, :, :, 9]
normalized_nx_elist = image_to_grid_graph(loaded_img)
G = nx.grid_2d_graph(loaded_img.shape[0], loaded_img.shape[1])
G.add_weighted_edges_from(normalized_nx_elist)
print_info(G)