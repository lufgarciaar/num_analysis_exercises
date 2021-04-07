from numpy import array, linalg, vstack
from scipy import optimize as opt
import sympy as sp
import random

from sympy.core.sympify import sympify

def steepest_descent(func, vars, /, start=None, *,
 err=0.001, max_iter=1000, xrange=(-1, 1), yrange=(-1, 1), full_output=False):
    """
    Compute Steepest DescentMethod.

    Parameters
    ----------
        func: sympy.core.mul.Mul 
            Expression with the function to minimize, returned by sympy. 
            x1 and x2 must be used as variables.
        vars: symbols
            Contains sympy symbols of the used variables in function.
        start: list, optional.
            Start point to search minimum. If not provided, a random point
            in xrange and yrange is selected.
        xrange: tuple, optional.
            Search range for the first x1 component.
        yrange: tuple, optional.
            Search range for the first x2 component.
        err: float, optional.
            Target error desired.
        max_iter: int, optional.
            Maximum number of iterations.
        full_output: boolean, optional.
            False: returns only minimum point.
            True: returns minimum and steps.
    Returns
    -------
        opt_point: ndarray
            Minimum point x1, x2.
        points: ndarray, optional.
            Every point evaluated by the algorithm.
    """
    
    x1, x2 = vars
    
    grad = sp.derive_by_array(func, [x1, x2])

    function = sp.lambdify([x1, x2], func, modules='numpy')
    gradient = sp.lambdify([x1, x2], grad, modules='numpy')

    if start is None:
        start_x = random.uniform(xrange[0], xrange[1])
        start_y = random.uniform(yrange[0], yrange[1])
        start = array([start_x, start_y])
    else:
        start = array(start)

    opt_boundaries = (0, 10)
    alpha = opt.fminbound(phi, *opt_boundaries, args=(function, gradient, start,))
    opt_point = start - alpha*array(gradient(*start))
    tol = scaled_difference(function, start, opt_point)

    if tol < err:
        stop_flag = True
    else:
        stop_flag = False

    points = start
    points = vstack((points, opt_point))
    prev_point = opt_point
    iter = 1

    while not (stop_flag):
        alpha = opt.fminbound(phi, *opt_boundaries, args=(function, gradient, prev_point,))
        opt_point = prev_point - alpha*array(gradient(*prev_point))

        tol = scaled_difference(function, prev_point, opt_point)
        if tol < err:
            stop_flag = True

        iter +=1
        points = vstack((points, opt_point))
        prev_point = opt_point

    if full_output:
        return opt_point, points
    else:
        return opt_point

def phi(alpha, function, gradient, point):
    eval_gradient = array(gradient(*point))
    arg = point-alpha*eval_gradient
    eval_phi = array(function(*arg))
    return eval_phi

def scaled_difference(func, prev_point, new_point):
    iter_difference = linalg.norm(array(func(*new_point))-array(func(*prev_point)))
    scl_diff = iter_difference/max(1, linalg.norm(array(func(*prev_point)))) 
    return scl_diff