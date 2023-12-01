from sympy import solve, symbols, init_printing

init_printing()

x, y, z = symbols('x, y, z')
r = symbols('r', constant=True)
x_v, y_v, z_v = symbols('x_v, y_v, z_v', constant=True)
x_f, y_f, z_f = symbols('x_f, y_f, z_f', constant=True)

equations = [
    (x - x_f)/(x_v -x_f) - (y - y_f)/(y_v - y_f), # on straight line between vertex and fish
    (x - x_f)/(x_v -x_f) - (z - z_f)/(z_v - z_f), # on straight line between vertex and fish
    x**2 +  z**2 - r**2 # on cylinder of radius r
]
solutions = solve(equations, [x,y,z], dict=True)

solutions[0][x]
solutions[1][x]