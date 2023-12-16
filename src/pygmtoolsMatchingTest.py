#!/usr/bin/env python3
# -- coding: utf-8 --
import numpy as np # numpy backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
pygm.set_backend('numpy') # set default backend for pygmtools
np.random.seed(1) # fix random seed

num_nodes2 = 10
A2 = np.random.rand(num_nodes2, num_nodes2)
A2 = (A2 + A2.T > 1.) * (A2 + A2.T) / 2
np.fill_diagonal(A2, 0)
n2 = np.array([num_nodes2])

num_nodes1 = 5
G2 = nx.from_numpy_array(A2)
pos2 = nx.spring_layout(G2)
pos2_t = np.array([pos2[_] for _ in range(num_nodes2)])
selected = [0] # build G1 as a cluster in visualization
unselected = list(range(1, num_nodes2))
while len(selected) < num_nodes1:
    dist = np.sum(np.sum(np.abs(np.expand_dims(pos2_t[selected], 1) - np.expand_dims(pos2_t[unselected], 0)), axis=-1), axis=0)
    select_id = unselected[np.argmin(dist).item()] # find the closest node from unselected
    selected.append(select_id)
    unselected.remove(select_id)
selected.sort()
A1 = A2[selected, :][:, selected]
X_gt = np.eye(num_nodes2)[selected, :]
n1 = np.array([num_nodes1])

G1 = nx.from_numpy_array(A1)
pos1 = {_: pos2[selected[_]] for _ in range(num_nodes1)}
color1 = ['#FF5733' for _ in range(num_nodes1)]
color2 = ['#FF5733' if _ in selected else '#1f78b4' for _ in range(num_nodes2)]
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Subgraph 1')
plt.gca().margins(0.4)
nx.draw_networkx(G1, pos=pos1, node_color=color1)
plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=pos2, node_color=color2)

print("end")

conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001) # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

plt.figure(figsize=(4, 4))
plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
plt.imshow(K, cmap='Blues')

X = pygm.rrwm(K, n1, n2)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('RRWM Soft Matching Matrix')
plt.imshow(X, cmap='Blues')
plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt, cmap='Blues')