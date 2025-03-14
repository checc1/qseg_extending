import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def draw(G: nx.Graph, image: np.ndarray) -> None:
  """
  Draw the graph G with the given image as node colors.
  :param G (networkx.Graph): Graph to be drawn.
  :param image (numpy.ndarray): Grayscale image for node colors.
  :return None
  """
  pixel_values = image
  fig, axs = plt.subplots(1, 2)
  #figsize=(min(12,2*image.shape[0]),min(12,2*image.shape[0])))
  #default_axes = plt.axes(frameon=True)
  pos = {(x,y):(y,-x) for x,y in G.nodes()}
  axs[0].imshow(image, plt.cm.Greys)
  nx.draw_networkx(G,
                  pos=pos,
                  node_color=1-pixel_values,
                  with_labels=False,
                  node_size=25,
                  cmap=plt.cm.Greys,
                  alpha=0.8,
                  ax=axs[1])
  nodes = nx.draw_networkx_nodes(G, pos, node_color=1-pixel_values,
                  node_size=25,
                  cmap=plt.cm.grey, ax=axs[1])
  nodes.set_edgecolor('tab:blue')
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G,
                               pos=pos,
                               edge_labels=edge_labels,
                               font_size=6)

