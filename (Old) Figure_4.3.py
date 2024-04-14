import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Set the seed for reproducibility
np.random.seed(4862)

# Constants and Psi value
n = 40
Psi = 0.10  # Psi is now constant
X = 5.0
q = 0.5
vmax = 1.0
phi_values = np.linspace(0, 1.69, 100)  # phi varies from 0 to 1.69

# Function to compute phi* for each value of phi
def phistar(phi, Psi):
    return 2 * Psi * phi

# Function to generate an adjusted matrix G
def generate_adjusted_matrix(n):
    G = np.zeros((n, n))
    possible_values = [0.5, -0.5, 1, 0]
    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i == j:
                continue
            if j == n - 1:
                G[i, j] = 1 - row_sum
            else:
                value = np.random.choice(possible_values)
                if row_sum + value > 1 or (j == n - 2 and row_sum + value + 0.5 > 1):
                    value = 1 - row_sum - 0.5 if np.random.rand() > 0.5 else 0
                G[i, j] = value
                row_sum += value
    return G

# Function to generate a diagonal matrix V
def generate_diagonal_matrix(n):
    possible_values_diagonal = [-1, 1, 0.5, -0.5]
    V = np.zeros((n, n))
    for i in range(n):
        V[i, i] = np.random.choice(possible_values_diagonal)
    return V

# Function to calculate pivotal probabilities
def calculate_pivotal_probabilities(x, n, Psi):
    pivotal_probs = np.zeros(n)
    for i in range(n):
        others = np.concatenate((x[:i], x[i+1:]))
        for j in range(n//2):
            pivotal_probs[i] += comb(n - 1, j) * np.prod(others[:j]) * np.prod(1 - others[j:])
    return pivotal_probs / (2**(n - 1))

# Function to solve the system for x*
def solve_system(n, phi, Psi, Gn):
    x = np.full(n, 0.5)  # Initial guess
    for _ in range(9999):  # Max iterations
        pivotal_probs = calculate_pivotal_probabilities(x, n, Psi)
        x_new = np.array([q + phistar(phi, Psi) * sum(Gn[i, :] * (1 - x)) for i in range(n)])
        if np.allclose(x, x_new, atol=1e-6):
            return x
        x = x_new
    raise ValueError("The system did not converge")

# Adjusted matrix G and diagonal matrix V
Gn = generate_adjusted_matrix(n)
V = generate_diagonal_matrix(n)

# Lists to store centrality measures for legislators 0 and 19
centrality_0 = []
centrality_19 = []

# Iterate over phi values and compute b_M for each, then store the required centralities
for phi in phi_values:
    x_star = solve_system(n, phi, Psi, Gn)
    V_diag = np.diag(V)
    matrix_inside_brackets = np.eye(n) - (phistar(phi, Psi) * Gn.T + Psi * np.diag(V_diag))
    inverted_matrix = np.linalg.inv(matrix_inside_brackets)
    b_M = inverted_matrix.dot(np.ones(n))
    centrality_0.append(b_M[0])
    centrality_19.append(b_M[19])

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(phi_values, centrality_0, label='Legislator 0 Centrality', marker='o')
plt.plot(phi_values, centrality_19, label='Legislator 19 Centrality', marker='o')
plt.title('Katz-Bonacich Centrality of Legislators 0 and 19 vs Phi')
plt.xlabel('Phi')
plt.ylabel('Katz-Bonacich Centrality Measure')
plt.legend()
plt.grid(True)
plt.savefig('centrality_vs_phi.png')  # Save the figure as a .png file
plt.show()