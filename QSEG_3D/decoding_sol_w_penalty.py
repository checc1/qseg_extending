import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df_sol = pd.read_csv("/Users/francescoaldoventurelli/qml/FeatureSelectionQubo/graph_image_qseg/solution_0.6.csv")
img_solution = list(df_sol.values)[:256]

plt.imshow(np.array(img_solution).reshape(16,16))
plt.title(r'Solution $\lambda$ = 0.6')
plt.show()