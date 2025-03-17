import h5py
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
    plt.title(f"Plane {k_plane}")
    plt.show()


def evolution_2d(img_idx: int) -> None:
    """
    Plot the evolution of a 2D image.
    :param img_idx: (int) Image index;
    :return: None
    """
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(xtrain[img_idx, :, :, 4 * i + j], cmap="viridis")
            axs[i, j].axis('off')
            axs[i, j].set_title(f"Plane {4 * i + j}")
    plt.show()


xtrain = x_train.reshape(x_train.shape[0], 16, 16, 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load the MNIST3D and plot 2D image")
    parser.add_argument("img_idx", type=int, help="Image index")
    parser.add_argument("ms", type=str, choices=["multi", "single"], help="Visualizing multi or single plane.")
    parser.add_argument("k_plane", type=int, nargs="?", default=None, help="Plane to visualize (only for 'single').")

    args = parser.parse_args()

    if args.ms == "multi":
        evolution_2d(args.img_idx)
        plt.show()
    else:
        if args.k_plane is None:
            parser.error("The 'single' mode requires specifying k_plane.")
        plot_2d(args.img_idx, args.k_plane)
        plt.show()
