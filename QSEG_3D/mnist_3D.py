import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse


path_to_data = "/Users/francescoaldoventurelli/Downloads/Mnist3D"
with h5py.File(path_to_data + '/full_dataset_vectors.h5', 'r') as dataset:
    x_train = dataset["X_train"][:]
    x_test = dataset["X_test"][:]
    y_train = dataset["y_train"][:]
    y_test = dataset["y_test"][:]


def plot_2d(img_idx: int, k_plane: int) -> None:
    """
    Plot a 2D image.
    :param img_idx: (int) Image index;
    :param k_plane: (int) Plane to visualize;
    :return: None
    """
    plt.imshow(xtrain[img_idx, :, :, k_plane], cmap="viridis")
    plt.show()

xtrain = x_train.reshape(x_train.shape[0], 16, 16, 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load the MNIST3D and plot 2D image")
    parser.add_argument("img_idx", type=int, help="Image index")
    parser.add_argument("k_plane", type=int, help="Plane to visualize")
    args = parser.parse_args()

    plot_2d(args.img_idx, args.k_plane)
    plt.show()
