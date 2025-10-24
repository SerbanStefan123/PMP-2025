from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

model = MarkovNetwork()
variables = ["A1", "A2", "A3", "A4", "A5"]
model.add_nodes_from(variables)

edges = [
    ("A1", "A2"),
    ("A1", "A3"),
    ("A2", "A4"),
    ("A2", "A5"),
    ("A3", "A4"),
    ("A4", "A5"),
]
model.add_edges_from(edges)

pos = nx.circular_layout(model)
plt.figure(figsize=(5, 5))
nx.draw(model, with_labels=True, pos=pos, node_size=1500, alpha=0.7)
plt.title("Markov Random Field (A1..A5)")
plt.show()


def pairwise_factor(var1, var2):
    values = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            values.append(np.exp(x + y))
    return DiscreteFactor(variables=[var1, var2], cardinality=[2, 2], values=values)


factors = []
for edge in edges:
    factors.append(pairwise_factor(edge[0], edge[1]))
model.add_factors(*factors)
print("Cliques of the model:")
print(model.get_cliques())
print("\nLocal independencies:")
print(model.get_local_independencies())

bp = BeliefPropagation(model)
map_query = bp.map_query(variables=variables)
print("\nMAP assignment (maximum probability configuration):")
print(map_query)
