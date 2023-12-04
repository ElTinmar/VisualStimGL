from sympy import solve, symbols, init_printing

init_printing()

x, y, z = symbols('x, y, z')
r = symbols('r', constant=True)
x_v, y_v, z_v = symbols('x_v, y_v, z_v', constant=True)
x_f, y_f, z_f = symbols('x_f, y_f, z_f', constant=True)

# intersect straight line between fish (x,y,z) and each vertex with cylinder.
# need two equations for the straight line, and one for the cylinder
equations = [
    (x - x_f)/(x_v -x_f) - (y - y_f)/(y_v - y_f), # on straight line between vertex and fish
    (x - x_f)/(x_v -x_f) - (z - z_f)/(z_v - z_f), # on straight line between vertex and fish
    x**2 +  z**2 - r**2 # on cylinder of radius r
]
solutions = solve(equations, [x,y,z], dict=True)

solutions[0]
solutions[1]

import numpy as np

x_v = -2
y_v = 0
z_v = -2

x_f = 0
y_f = 0
z_f = -5

r = 100

denominator = (x_f*x_f - 2*x_f*x_v + x_v*x_v + z_f*z_f - 2*z_f*z_v + z_v*z_v)
squareroot = np.sqrt(r*r*x_f*x_f - 2*r*r*x_f*x_v + r*r*x_v*x_v + r*r*z_f*z_f - 2*r*r*z_f*z_v + r*r*z_v*z_v - x_f*x_f*z_v*z_v + 2*x_f*x_v*z_f*z_v - x_v*x_v*z_f*z_f)
x0 = 1/denominator * (-x_f*z_f*z_v + x_f*z_v*z_v - x_f*squareroot + x_v*z_f*z_f - x_v*z_f*z_v + x_v*squareroot)
x1 = 1/denominator * (-x_f*z_f*z_v + x_f*z_v*z_v + x_f*squareroot + x_v*z_f*z_f - x_v*z_f*z_v - x_v*squareroot)
y0 = 1/denominator * (x_f*x_f*y_v + x_f*x_v*y_f - x_f*x_v*y_v + x_v*x_v*y_f - y_f*z_f*z_v + y_f*z_v*z_v - y_f*squareroot + y_v*z_f*z_f - y_v*z_f*z_v + y_v*squareroot)
y1 = 1/denominator * (x_f*x_f*y_v + x_f*x_v*y_f - x_f*x_v*y_v + x_v*x_v*y_f - y_f*z_f*z_v + y_f*z_v*z_v + y_f*squareroot + y_v*z_f*z_f - y_v*z_f*z_v - y_v*squareroot)
z0 = 1/denominator * ((x_f - x_v)*(x_f*z_v - x_v*z_f) - (z_f - z_v) * squareroot)
z1 = 1/denominator * ((x_f - x_v)*(x_f*z_v - x_v*z_f) + (z_f - z_v) * squareroot)

sol0 = np.array([x0,y0,z0])
sol1 = np.array([x1,y1,z1])
fish = np.array([x_f,y_f,z_f])
vertex = np.array([x_v,y_v,z_v])

np.dot(sol0-fish,vertex-fish)
np.dot(sol1-fish,vertex-fish)