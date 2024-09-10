import numpy as np
import math
import casadi

# z = [u, x]
# u = [ax, ay, az]
# x = [pos_x, pos_y, pos_z, vx, vy, vz]
def second_order_dynamics(u, x, dt = 0.1):
    dim = 2
    integral = np.zeros(dim*2)
    integral[:dim] = 0.5 * dt**2 * u
    integral[dim:] = dt * u

    phi = np.eye(dim*2)
    for i in range(dim):
        phi[i, i+dim] = dt

    return np.dot(phi, x) + integral


def second_order_dynamics_casadi(u, x, dt=0.1):
    """
    using casadi to implement second order dynamics
    """
    dim = 2
    integral = casadi.SX.zeros(dim * 2)
    integral[:dim] = 0.5 * dt ** 2 * u
    integral[dim:] = dt * u

    phi = casadi.SX.eye(dim * 2)
    for i in range(dim):
        phi[i, i + dim] = dt

    return casadi.mtimes(phi, x) + integral



def first_order_dynamics(u, x, dt = 0.1):
    """
    u = [vx1,vy1, vx2, vy2, ...]
    x = [x1, y1, x2, y2, ...]
    """
    return x + u * dt