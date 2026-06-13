from einmesh.numpy import einmesh

from einmesh import (
    LinSpace,
    LogSpace,
    UniformDistribution,
)

import einx
import numpy as np


#creating the same 2d meshgrid in einmesh, einx and numpy 
xs, ys = LinSpace(-2, 2, 5), LinSpace(0, 1, 3)
x_coords, y_coords = einmesh("y x", x = xs, y = ys)

print(x_coords)
print(y_coords)

xs, ys = np.linspace(-2, 2, 5), np.linspace(0, 1, 3)

x, y = einx.id("x, y-> y x, y x", xs, ys)
print(x)
print(y)

x, y = np.meshgrid(xs, ys)
print(x)
print(y)


#return a single tensor by stacking

xs, ys = LinSpace(-2, 2, 5), LinSpace(0, 1, 3)
coords = einmesh("x * y", x = xs, y = ys)
print(coords)

xs, ys = np.linspace(-2, 2, 5), np.linspace(0, 1, 3)

coords = einx.id("x, y-> x (1+1) y", xs, ys)
print(coords)



#stacking in einx

x = np.arange(12).reshape(3, 4)
y = np.arange(9).reshape(3, 3)

print(einx.id("a b, a c-> a (b + c)", x, y))


#3d

xs, ys, zs = np.linspace(-2, 2, 5), np.linspace(0, 1, 3), np.linspace(0, 4, 3)

x = einx.id("x, y, z-> (1+1+1) x y z", xs, ys, zs)
print(x)
