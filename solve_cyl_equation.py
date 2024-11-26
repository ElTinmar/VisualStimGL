from sympy import solve, symbols, init_printing
import numpy as np

# solving

init_printing()

x, y, z = symbols('x, y, z')
r = symbols('r', constant=True)
x_v, y_v, z_v = symbols('x_v, y_v, z_v', constant=True)
x_f, y_f, z_f = symbols('x_f, y_f, z_f', constant=True)

# intersect straight line between fish (x,y,z) and each vertex with cylinder.
# need two equations for the straight line, and one for the cylinder
equations = [
    (x - x_f)/(x_v - x_f) - (y - y_f)/(y_v - y_f), # on straight line between vertex and fish
    (x - x_f)/(x_v - x_f) - (z - z_f)/(z_v - z_f), # on straight line between vertex and fish
    x**2 +  z**2 - r**2 # on cylinder of radius r
]
solutions = solve(equations, [x,y,z], dict=True)

solutions[0]
solutions[1]

# implementation

def solutions0(v, f, r):

    # vertex coords
    x_v = v[0]
    y_v = v[1]
    z_v = v[2]

    # fish coords
    x_f = f[0]
    y_f = f[1]
    z_f = f[2]

    # (x_f-x_v)**2 + (z_f-z_v)**2
    denominator = (
        + x_f*x_f 
        - 2*x_f*x_v 
        + x_v*x_v 
        + z_f*z_f 
        - 2*z_f*z_v 
        + z_v*z_v
    )

    # r**2 * [ (x_f-x_v)**2 + (z_f-z_v)**2 ] - (x_f*z_v - x_v*z_f)**2 
    squareroot = np.sqrt(
        + r*r*x_f*x_f 
        - 2*r*r*x_f*x_v 
        + r*r*x_v*x_v 
        + r*r*z_f*z_f 
        - 2*r*r*z_f*z_v 
        + r*r*z_v*z_v 
        - x_f*x_f*z_v*z_v 
        + 2*x_f*x_v*z_f*z_v 
        - x_v*x_v*z_f*z_f
    )

    # x_v * [z_f*(z_f-z_v) + squareroot] - x_f * [z_v*(z_f-z_v) + squareroot]
    x0 = 1/denominator * (
        - x_f*z_f*z_v 
        + x_f*z_v*z_v 
        - x_f*squareroot 
        + x_v*z_f*z_f 
        - x_v*z_f*z_v 
        + x_v*squareroot
    )

    # x_v * [z_f*(z_f-z_v) - squareroot] - x_f * [z_v*(z_f-z_v) - squareroot]
    x1 = 1/denominator * (
        - x_f*z_f*z_v 
        + x_f*z_v*z_v 
        + x_f*squareroot 
        + x_v*z_f*z_f 
        - x_v*z_f*z_v 
        - x_v*squareroot
    )

    # (x_f - x_v)*(x_f*y_v - x_v*y_f) + y_v * [z_f*(z_f-z_v) + squareroot] - y_f * [z_v*(z_f-z_v) + squareroot]
    y0 = 1/denominator * (
        + x_f*x_f*y_v 
        - x_f*x_v*y_f 
        - x_f*x_v*y_v 
        + x_v*x_v*y_f 

        - y_f*z_f*z_v 
        + y_f*z_v*z_v 
        - y_f*squareroot 

        + y_v*z_f*z_f 
        - y_v*z_f*z_v 
        + y_v*squareroot
    )

    # (x_f - x_v)*(x_f*y_v - x_v*y_f) + y_v * [z_f*(z_f-z_v) - squareroot] - y_f * [z_v*(z_f-z_v) - squareroot]
    y1 = 1/denominator * (
        + x_f*x_f*y_v 
        - x_f*x_v*y_f 
        - x_f*x_v*y_v 
        + x_v*x_v*y_f 

        - y_f*z_f*z_v 
        + y_f*z_v*z_v 
        + y_f*squareroot 

        + y_v*z_f*z_f 
        - y_v*z_f*z_v 
        - y_v*squareroot
    )

    z0 = 1/denominator * (
        + (x_f - x_v)*(x_f*z_v - x_v*z_f) 
        - (z_f - z_v) * squareroot
    )
    z1 = 1/denominator * (
        + (x_f - x_v)*(x_f*z_v - x_v*z_f) 
        + (z_f - z_v) * squareroot
    )

    sol0 = np.array([x0,y0,z0])
    sol1 = np.array([x1,y1,z1])
    return (sol0, sol1)

def solutions1(v, f, r):

    # vertex coords
    x_v = v[0]
    y_v = v[1]
    z_v = v[2]

    # fish coords
    x_f = f[0]
    y_f = f[1]
    z_f = f[2]

    # helpful variables;
    x_ = x_f-x_v
    z_ = z_f-z_v
    xz_ = (x_f*z_v - x_v*z_f)
    xy_ = (x_f*y_v - x_v*y_f)
    d = x_*x_ + z_*z_
    s = np.sqrt(r*r*d - xz_*xz_)

    # projection to cylinder
    x0 = 1/d * (x_v*(z_f*z_ + s) - x_f*(z_v*z_ + s))
    x1 = 1/d * (x_v*(z_f*z_ - s) - x_f*(z_v*z_ - s))
    y0 = 1/d * (x_*xy_ + y_v * (z_f*z_ + s) - y_f * (z_v*z_ + s))
    y1 = 1/d * (x_*xy_ + y_v * (z_f*z_ - s) - y_f * (z_v*z_ - s))
    z0 = 1/d * (x_*xz_ - z_*s)
    z1 = 1/d * (x_*xz_ + z_*s)

    sol0 = np.array([x0,y0,z0])
    sol1 = np.array([x1,y1,z1])
    return (sol0, sol1)

solutions0([-2,1,5],[1,6,-2],10)
solutions1([-2,1,5],[1,6,-2],10)