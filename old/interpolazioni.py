# 3D demo: Linear vs Spherical (SLERP) interpolation between two vectors on a unit sphere
import numpy as np
import matplotlib.pyplot as plt

# --- helpers ---
def slerp(p0, p1, t_array):
    p0_n = p0 / np.linalg.norm(p0)
    p1_n = p1 / np.linalg.norm(p1)
    dot = np.clip(np.dot(p0_n, p1_n), -1.0, 1.0)
    theta = np.arccos(dot)
    if np.isclose(theta, 0):
        return np.outer(1 - t_array, p0_n) + np.outer(t_array, p1_n)
    s1 = np.sin((1 - t_array) * theta) / np.sin(theta)
    s2 = np.sin(t_array * theta) / np.sin(theta)
    return (s1[:, None] * p0_n[None, :]) + (s2[:, None] * p1_n[None, :])

def lerp(p0, p1, t_array):
    return (1 - t_array)[:, None] * p0[None, :] + t_array[:, None] * p1[None, :]

# pick two random points on the unit sphere
rng = np.random.default_rng(0)
a = rng.normal(size=3)
b = rng.normal(size=3)
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)

t = np.linspace(0, 1, 200)
path_slerp = slerp(a, b, t)
path_lerp = lerp(a, b, t)

# Create a sphere wireframe
phi = np.linspace(0, np.pi, 30)
theta = np.linspace(0, 2*np.pi, 30)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# --- plot ---
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# sphere surface (light wireframe)
ax.plot_wireframe(x, y, z, linewidth=0.3, alpha=0.3)

# endpoints
ax.scatter([a[0]], [a[1]], [a[2]], s=50, label='z1')
ax.scatter([b[0]], [b[1]], [b[2]], s=50, label='z2')

# paths
print(a)
print(b)
ax.plot(path_lerp[:,0], path_lerp[:,1], path_lerp[:,2], linewidth=2, label='LERP (line)')
ax.plot(path_slerp[:,0], path_slerp[:,1], path_slerp[:,2], linewidth=2, label='SLERP (arc)')

# aesthetics
ax.set_box_aspect([1,1,1])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend(loc='upper left')
ax.set_title('Linear vs Spherical Interpolation on the Unit Sphere')

plt.show()
