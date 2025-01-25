import torch 
import cvxpy as cp
import numpy as np

def mosek_solve(A,weight,c,n):
    """
    examples 1  weighted square root utility: 
                F = cp.Variable(A.shape) 
                bimask = np.ones((n,n))
                np.fill_diagonal(bimask, 0)
                obj = -cp.sum(cp.multiply(weight,cp.sqrt(-cp.multiply(bimask,F@A.T))))
                prob = cp.Problem(cp.Minimize(obj),[F>=0,F.T@np.ones(n)<=c])
                prob.solve(solver=cp.MOSEK)
                cvx_optimal = prob.value.item()
                mosek_time = prob._solve_time
                return mosek_time, cvx_optimal
    example 2   weighted log utility:
                F = cp.Variable(A.shape) 
                bimask = np.ones((n,n))
                np.fill_diagonal(bimask, 0)
                obj = -cp.sum(cp.multiply(weight,cp.log(-cp.multiply(bimask,F@A.T)+np.eye(n))))
                prob = cp.Problem(cp.Minimize(obj),[F>=0,F.T@np.ones(n)<=c])
                prob.solve(solver=cp.MOSEK)
                cvx_optimal = prob.value
                mosek_time = prob._solve_time
                return mosek_time, cvx_optimal
    """
    pass

def prox_util(V, beta, weight):
    """
    examples 1  weighted square root utility: 
                b = -V.to(torch.complex64)
                d = (beta * weight**2 / 4).to(torch.complex64)
                del_1 = 2 * b**3 + 27 * d
                C = ((del_1 + (del_1**2 - 4 * b**6)**0.5) / 2)**(1/3)
                res = torch.real(- (b + C + b**2 / C)/3)
                res.fill_diagonal_(0)
                return res
    example 2   weighted log utility:
                res = (V - (V**2 + 4 * beta * weight)**0.5)/2
                res.fill_diagonal_(0)
                return res
    """
    pass

def eval_f(minus_FAt, weight):
    """ 
    example 1   weighted square root utility:
                minus_FAt.fill_diagonal_(0)
                return ((- weight * minus_FAt**0.5).sum())
    example 2   weighted log utility:
                minus_FAt.fill_diagonal_(1)
                return ((- weight * torch.log(minus_FAt)).sum())
    """
    pass 

def nabla(minus_FAt, weight):
    """
    example 1   weighted square root utility:
                inv_minus_FAt = - weight / (2 * minus_FAt**0.5)
                inv_minus_FAt.fill_diagonal_(0)
                return inv_minus_FAt
    example 2   weighted log utility:
                minus_FAt.fill_diagonal_(1)
                inv_minus_FAt = (1 / minus_FAt) * weight
                inv_minus_FAt.fill_diagonal_(0)
                return inv_minus_FAt

    """
    pass 