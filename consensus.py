from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# General properties
t0 = 0
tf = 10
dt = 0.1
T = int((tf - t0) / dt)

# Define the points
x0 = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [0.5, np.sqrt(3) / 2, np.sqrt(3) / 2]])

x = np.zeros((T, 3, 3))
x[0, :, :] = x0
xproj = np.zeros((T, 3, 3))
xproj[0, :, :] = x0
r = np.array([[-1.5, -np.sqrt(3) / 2, 0],
              [1.5, -np.sqrt(3) / 2, 0],
              [0, np.sqrt(3) , 0]])
xref = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [0.5, np.sqrt(3) / 2, 0]])

# Refereence points
# x_ref = np.array([])


# Laplacian
L = np.array([[2, -1, -1],
              [-1, 2, -1],
              [-1, -1, 2]])

for t in range(T - 1):
    # x1dot = - ((x[t, 0, :] - xref[0, :] - x[t, 1, :] + xref[1, :]) + (
    #         x[t, 0, :] - xref[0, :] - x[t, 2, :] + xref[2, :]))
    # x2dot = - ((x[t, 1, :] - xref[1, :] - x[t, 0, :] + xref[0, :]) + (
    #         x[t, 1, :] - xref[1, :] - x[t, 2, :] + xref[2, :]))
    # x3dot = - ((x[t, 2, :] - xref[2, :] - x[t, 1, :] + xref[1, :]) + (
    #         x[t, 2, :] - xref[2, :] - x[t, 0, :] + xref[0, :]))
    # x1dot = - ((x[t, 0, :]  - x[t, 1, :] ) + (x[t, 0, :]  - x[t, 2, :])) + r[0,:]
    # x2dot = - ((x[t, 1, :] - x[t, 0, :]) + (x[t, 1, :] - x[t, 2, :])) + r[1, :]
    # x3dot = - ((x[t, 2, :] - x[t, 1, :]) + (x[t, 2, :] - x[t, 0, :])) + r[2, :]
    # x[t + 1, 0, :] = x1dot * dt + x[t, 0, :]
    # x[t + 1, 1, :] = x2dot * dt + x[t, 1, :]
    # x[t + 1, 2, :] = x3dot * dt + x[t, 2, :]
    xdot = -L @ (x[t, :, :]-xref)
    xdot = -L @ (x[t, :, :]) + r
    x[t + 1, :, :] = x[t, :, :] + xdot * dt

# Create the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the triangle surface
triangle0 = Poly3DCollection([x0], alpha=0.5, edgecolor='g')
triangle0.set_facecolor('g')  # You can change the color
triangleref = Poly3DCollection([xref], alpha=0.5, edgecolor='r')
triangleref.set_facecolor('b')  # You can change the color
# Add the triangle to the plot
ax.add_collection3d(triangle0)
ax.add_collection3d(triangleref)
# Plot the vertices
ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], color='g')
ax.scatter(xref[:, 0], xref[:, 1], xref[:, 2], color='b')
# Set axes limits (optional for better view)
ax.set_xlim(0, 1)
ax.set_ylim(0, 2)
ax.set_zlim(0, 2)

# Labels (optional)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for t in range(T):
    # Create the triangle surface
    triangle = Poly3DCollection([x[t, :, :]], alpha=0.5, edgecolor='r')
    triangle.set_facecolor('r')  # You can change the color
    # Add the triangle to the plot
    ax.add_collection3d(triangle)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])
    plt.pause(1)
    triangle.remove()
    #
    # plt.clf()
