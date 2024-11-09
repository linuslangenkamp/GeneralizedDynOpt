from sympy import *
from numpy.polynomial.legendre import leggauss
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import legendre
import numpy as np

flip = lambda x: 1 - (x + 1) / 2


def radau_iia_nodes(s):
    """
    Compute the Radau IIA nodes by finding the roots of the polynomial P_{N-1}(x) + P_N(x).
    Args:
        s (int): Number of stages.
    Returns:
        nodes (ndarray): The Radau nodes in the interval [0, 1].
    """

    # Define the polynomials P_{N-1} and P_N
    P_N_minus_1 = legendre(s - 1)
    P_N = legendre(s)

    # Form the polynomial P_{N-1}(x) + P_N(x)
    radau_poly = P_N_minus_1 + P_N

    # Find the roots in [0, 1] of P_{N-1}(x) + P_N(x)
    nodes = np.roots(radau_poly.coefficients)
    nodes = [flip(n) for n in nodes]
    # Append 1 as the last node for Radau IIA
    nodes = np.sort(np.append(nodes, 1))

    return nodes[:-1]


def radau_iia_butcher_tableau(s):
    """
    Compute the Butcher tableau for Radau IIA method with s stages.
    Args:
        s (int): Number of stages
    Returns:
        A (Matrix): The A matrix in the Butcher tableau
        b (Matrix): The b vector in the Butcher tableau
    """
    # Compute Radau nodes: roots of the shifted Legendre polynomial derivative
    nodes = radau_iia_nodes(s)
    print(nodes)
    # Set up symbolic variables for polynomials
    t = symbols("t")
    lagrange_basis = []
    A = np.zeros((s, s))

    # Compute Lagrange polynomials
    for j, cj in enumerate(nodes):
        lj = 1
        for m, cm in enumerate(nodes):
            if m != j:
                lj *= (t - cm) / (cj - cm)
        lagrange_basis.append(lj)

    # Compute the A matrix
    for i in range(s):
        for j in range(s):
            A[i, j] = integrate(lagrange_basis[j], (t, 0, nodes[i]))

    # The weights b are the last row of A
    b = Matrix(A[-1, :])

    return Matrix(A), b


# Generate Radau IIA Butcher tableaus for stages 1 to 5
butcher_tableaus = [{"A": Matrix([[1]]), "b": Matrix([1])}] + [
    {"A": radau_iia_butcher_tableau(s)[0], "b": radau_iia_butcher_tableau(s)[1]}
    for s in range(2, 6)
]

z = Symbol("z")
# Define the center and side length for the square plot
center_x, center_y = 5, 0  # Custom center
side_length = 38  # Custom side length
half_side = side_length / 2

# Generate the grid of complex numbers
realVals = np.linspace(center_x - half_side, center_x + half_side, 500)
imagVals = np.linspace(center_y - half_side, center_y + half_side, 500)
Zreal, Zimag = np.meshgrid(realVals, imagVals)
Z = Zreal + 1j * Zimag

# Initialize the plot
plt.figure(figsize=(10, 8))

# Loop over each Butcher tableau to calculate and plot the stability region
for tableau in butcher_tableaus:
    A = tableau["A"]
    b = tableau["b"]
    ones = Matrix([1] * len(b))

    # Compute the stability function R(z) = 1 + z * b.T * (I - zA)^(-1) * ones
    inverse = (eye(len(b)) - z * A).inv()
    Rz = 1 + z * (b.T * (inverse * ones))[0]
    Rz = Rz.simplify()
    print(Rz)
    # Convert the symbolic R(z) to a numerical function
    R = lambdify(z, Rz, "numpy")

    # Compute |R(z)| for each point in the grid
    Rvals = R(Z)
    Rabs = np.abs(Rvals)

    # Plot the contour for |R(z)| = 1
    plt.contour(Zreal, Zimag, Rabs, levels=[1], colors="black", linewidths=1.5)

    # Shade the region where |R(z)| <= 1
    # Shade the region where |R(z)| <= 1 with a darker gray color
    plt.contourf(Zreal, Zimag, Rabs, levels=[0, 1], colors=["#a9a9a9"], alpha=0.35)

label_positions = [
    (2.5, 1.5),
    (6, 4),
    (10.5, 7),
    (15.5, 11),
    (21, 16),
]  # Adjust positions as needed
labels = [
    f"m = {i+1}" for i in range(len(label_positions))
]  # Labels for each stability region

for i, tableau in enumerate(butcher_tableaus):
    if i < len(label_positions):  # Ensure we have a label position for each contour
        x_label, y_label = label_positions[i]
        plt.text(
            x_label,
            y_label,
            labels[i],
            fontsize=10,
            ha="center",
            va="center",
            color="black",
        )


# Plot settings
plt.xlabel("Real $z$", fontsize=18)
plt.ylabel("Imag $z$", fontsize=18)
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")

# Set equal aspect ratio with custom limits based on center and side length
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(center_x - half_side, center_x + half_side)
plt.ylim(center_y - half_side, center_y + half_side)

plt.show()
