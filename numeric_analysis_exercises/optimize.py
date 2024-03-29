import random

import sympy as sp
from numpy import array, linalg, vstack
from scipy import optimize as opt
from sympy.core.sympify import sympify


def steepest_descent(func, sym_vars, /, start=(0, 0), *,
                     err=0.001, max_iter=1000, stop_criteria='sc_diff', step_type='opt', step=0.1, full_output=False):
    """
    Compute Steepest DescentMethod.

    Parameters
    ----------
        func: sympy.core.mul.Mul
            Expression with the function to minimize, returned by sympy.
            x1 and x2 must be used as variables.
        sym_vars: symbols
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
        stop_criteria: {'sc_diff', 'gradient', 'abs_diff'}, optional.
            'sc_diff':
                Check if the normalized func evaluated in the last point
                is less than err.
            'gradient':
                Check if the norm of the gradient in the last point is
                less than err.
            'abs_diff':
                Check if the difference between the function evaluated
                in the previous and the last point is less than err.
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

    if not stop_criteria in ['sc_diff', 'gradient', 'abs_diff']:
        raise ValueError(
            "stop_criteria is one of 'sc_diff', 'gradient', 'abs_diff'")

    if not step_type in ['opt', 'fix']:
        raise ValueError("step_type is one of 'opt', 'fix'")

    grad = sp.derive_by_array(func, [*sym_vars])

    function = sp.lambdify([*sym_vars], func, modules='numpy')
    gradient = sp.lambdify([*sym_vars], grad, modules='numpy')

    start = array(start)

    opt_boundaries = (0, 10)

    if step_type == 'opt':
        alpha = opt.fminbound(phi, *opt_boundaries,
                              args=(function, gradient, start,))
    if step_type == 'fix':
        alpha = step

    opt_point = start - alpha*array(gradient(*start))
    stop_flag = term_criterion(
        function, start, opt_point, criteria=stop_criteria, err=err)

    points = start
    points = vstack((points, opt_point))
    prev_point = opt_point
    iter = 1

    while not (stop_flag):
        if step_type == 'opt':
            alpha = opt.fminbound(phi, *opt_boundaries,
                                  args=(function, gradient, prev_point,))
        if step_type == 'fix':
            alpha = step
        opt_point = prev_point - alpha*array(gradient(*prev_point))
        stop_flag = term_criterion(
            function, prev_point, opt_point, criteria=stop_criteria, err=err)

        iter += 1
        points = vstack((points, opt_point))
        prev_point = opt_point

        if iter >= max_iter:
            stop_flag = True

    if full_output:
        return opt_point, points
    else:
        return opt_point


def phi(alpha, function, gradient, point):
    eval_gradient = array(gradient(*point))
    arg = point-alpha*eval_gradient
    eval_phi = array(function(*arg))
    return eval_phi


def term_criterion(function, prev_point, new_point, *, criteria='sc_diff', err=0.001):

    if criteria == 'sc_diff':
        tol = scaled_difference(function, prev_point, new_point)

    if criteria == 'gradient':
        tol = grad_criteria(function, new_point)

    if criteria == 'abs_diff':
        tol = absolute_difference(function, prev_point, new_point)

    if tol < err:
        criteria_flag = True
    else:
        criteria_flag = False

    return criteria_flag


def scaled_difference(func, prev_point, new_point):
    iter_difference = linalg.norm(
        array(func(*new_point))-array(func(*prev_point)))
    scl_diff = iter_difference/max(1, linalg.norm(array(func(*prev_point))))
    return scl_diff


def grad_criteria(grad, point):
    eval_grad = array(grad(*point))
    grad_norm = linalg.norm(eval_grad)
    return grad_norm


def absolute_difference(func, prev_point, new_point):
    iter_difference = linalg.norm(
        array(func(*new_point))-array(func(*prev_point)))
    return iter_difference
