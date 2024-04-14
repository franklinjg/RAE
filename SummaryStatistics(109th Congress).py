
import numpy as np
import pandas as pd
from sympy import Matrix
import scipy.io
from scipy.optimize import minimize
from scipy.special import comb, logsumexp
import matplotlib.pyplot as plt
import networkx as nx
import requests
import pyreadr
import os
import tempfile
from scipy import stats


def store_matrix_entries(matrix):
    stored_entries = {}
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            identifier = f"g_{i+1}{j+1}"
            stored_entries[identifier] = matrix[i, j]
    return stored_entries

# URL of the raw R data file on GitHub
url = 'https://raw.githubusercontent.com/franklinjg/RAE/29da2338741e37a621442490179455c570402bf4/G_party.rda'

# Make a request to get the content of the .rda file
response = requests.get(url)
if response.status_code == 200:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    # Read the R data file from the temporary file
    result = pyreadr.read_r(tmp_file_path)

    # Optionally delete the temporary file if you're done with it
    os.remove(tmp_file_path)

    # Now you can use 'result' as a normal result object from pyreadr
    print(result.keys())  # For example, to print the keys of the loaded R data

    # Assuming 'G_party' is the key for the data in the .rda file
    G_party = result['G_party']

    # Extract the 429x429 block from the DataFrame and convert it to a numpy matrix
    Gn = G_party.iloc[0:429, 0:429].values

    # Transpose the numpy matrix G
    G_transpose = Gn.T

    # Storing matrix entries in a dictionary from the extracted block
    entries_dict = store_matrix_entries(Gn)
else:
    print("Failed to fetch the file: Status code", response.status_code)



def graph_summary_statistics(matrix):
    G = nx.from_numpy_array(matrix)
    
    num_nodes = G.number_of_nodes()  # Total number of nodes
    num_edges = G.number_of_edges()  # Total number of edges
    density = nx.density(G)  # Density of the graph
    degrees = [deg for node, deg in G.degree()]  # List of degrees of all nodes
    average_degree = np.mean(degrees)  # Average degree
    max_degree = np.max(degrees)  # Maximum degree
    min_degree = np.min(degrees)  # Minimum degree

    return {
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Density of the Graph": density,
        "Average Degree": average_degree,
        "Maximum Degree": max_degree,
        "Minimum Degree": min_degree,
        "Degree Distribution": degrees
    }


stats = graph_summary_statistics(Gn)
print(stats)
