import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def function1(x: npt.NDArray[np.float64]) -> float:
    return (x[0] - 1)**2 + (x[1] - 2)**2

def function2(x: npt.NDArray[np.float64]) -> float:
    return (
        20
        + x[0]**2
        + x[1]**2
        - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    )

def function3(x: npt.NDArray[np.float64]) -> float:
    return (
        (x[0] - 1)**2
        + (x[1] - 2)**2
        + 2.5 * np.sin(x[0]) * np.sin(x[1])
    )



# Grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)

X, Y = np.meshgrid(x, y)

Z = np.vectorize(lambda a, b: function3(np.array([a, b])))(X, Y)

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

plt.show()