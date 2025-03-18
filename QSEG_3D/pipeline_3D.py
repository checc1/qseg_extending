import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from convert_to_2d_gridgraph import image_to_grid_graph
from solve_mincut import annealer_solver
from mnist_3D import xtrain as X
from decoding import decode_binary_string
from tqdm import tqdm


def plot(original_slices: list[np.array], mask_list: list[np.array]) -> None:
    """
    Plot the comparison between the original 2D image slices and the corresponding segmented masks.
    :param original_slices: (list[np.array]) Original 2D slices from the 3D image.
    :param mask_list: (list[np.array]) Corresponding segmentation masks.
    :return: None
    """
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    for i in range(4):
        for j in range(4):
            k = 4 * i + j
            axs[i, 2 * j].imshow(original_slices[k], cmap="viridis")
            axs[i, 2 * j].axis('off')
            axs[i, 2 * j].set_title(f"Original {k}")

            axs[i, 2 * j + 1].imshow(mask_list[k], cmap="gray")
            axs[i, 2 * j + 1].axis('off')
            axs[i, 2 * j + 1].set_title(f"Segmented {k}")

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Solve the MinCut problem on a 3D image using a D-Wave annealer."
                                     "The image is a 3D numpy array which is run along the common"
                                     "z axis and a 2D Q-Seg. segmentation algorithm is performed.")
    parser.add_argument("index", type=int, help="Index of the single image.")
    parser.add_argument("runs", type=int, help="Number of reads for the sampler.")
    args = parser.parse_args()

    x_img = X[args.index, :, :, :]
    solution_list = []

    for k in tqdm(range(x_img.shape[-1])): ## along z direction
        normalized_nx_elist = image_to_grid_graph(x_img[:, :, k])
        G = nx.grid_2d_graph(x_img.shape[0], x_img.shape[1])
        G.add_weighted_edges_from(normalized_nx_elist)
        samples_dataframe, execution_info_dict = annealer_solver(G, n_samples=args.runs)
        solution_binary_string = samples_dataframe.iloc[0][:-3]
        height, width, _ = x_img.shape
        segmentation_mask = decode_binary_string(solution_binary_string[:height * width], height, width)
        solution_list.append(np.array(segmentation_mask))

    plot([x_img[:, :, k] for k in range(x_img.shape[-1])], solution_list)