import time
import dimod
import neal
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse
from decoding import decode_binary_string
from qiskit_optimization.applications import Maxcut
from convert_to_2d_gridgraph import G, loaded_img as img
import qiskit_optimization.applications as applications


def dwave_solver(linear: applications, quadratic: applications, runs: int, **kwargs):
  """
  Solve a binary quadratic model using D-Wave sampler.

  Parameters:
  linear (dict): Linear coefficients of the model.
  quadratic (dict): Quadratic coefficients of the model.
  private_token (str): API token for D-Wave.
  runs (int): Number of reads for the sampler.

  Returns:
  dimod.SampleSet: Sample set returned by D-Wave sampler.
  float: Connection time.
  float: Embedding time.
  float: Response time.
  """
  vartype = dimod.BINARY
  sampler = neal.SimulatedAnnealingSampler()
  bqm_original = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype)
  start_time = time.time()
  sample_set = sampler.sample(bqm_original, num_reads=runs)
  response_time = time.time() - start_time

  return sample_set, 0, 0, response_time



def annealer_solver(G: nx.Graph, n_samples: int, **kwargs):
  """
  Solve the Maxcut problem on graph G using a D-Wave annealer.

  Parameters:
  G (networkx.Graph): Graph for which Maxcut is to be solved.
  private_token (str): API token for D-Wave.
  n_samples (int): Number of samples to collect.

  Returns:
  pandas.DataFrame: Dataframe containing samples.
  dict: Dictionary containing information about execution times.
  """
  start_time = time.time()
  w = -1 * nx.adjacency_matrix(G).todense()
  max_cut = Maxcut(w)
  qp = max_cut.to_quadratic_program()
  linear = qp.objective.linear.coefficients.toarray(order=None, out=None)
  quadratic = qp.objective.quadratic.coefficients.toarray(order=None, out=None)
  linear = {int(idx): round(value, 4) for idx, value in enumerate(linear[0])}
  quadratic = {(int(iy), int(ix)): -quadratic[iy, ix] for iy, ix in np.ndindex(quadratic.shape) if
               iy < ix and abs(quadratic[iy, ix]) != 0}

  problem_formulation_time = time.time() - start_time
  sample_set, connection_time, embedding_time, response_time = dwave_solver(linear, quadratic,
                                                                            runs=n_samples)
  info_dict = sample_set.info['timing'].copy()

  start_time = time.time()
  samples_df = sample_set.to_pandas_dataframe()
  sample_fetch_time = time.time() - start_time

  info_dict['problem_formulation_time'] = problem_formulation_time
  info_dict['connection_time'] = connection_time
  info_dict['embedding_time'] = embedding_time
  info_dict['response_time'] = response_time
  info_dict['sample_fetch_time'] = sample_fetch_time

  return samples_df, info_dict


if __name__ == "__main__":
      parser = argparse.ArgumentParser(description='Solve the MinCut problem on a medical image.')
      parser.add_argument("runs", type=int, help="Number of reads for the sampler.")
      args = parser.parse_args()

      samples_dataframe, execution_info_dict = annealer_solver(G, n_samples=args.runs)
      solution_binary_string = samples_dataframe.iloc[0][:-3]
      height, width = img.shape
      print(len(solution_binary_string))
      segmentation_mask = decode_binary_string(solution_binary_string[:height*width], height, width)
      fig, axs = plt.subplots(1, 2)
      axs[0].imshow(img, cmap='gray')
      axs[1].imshow(segmentation_mask, cmap='gray')
      plt.show()
