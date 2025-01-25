import numpy as np 
import cvxpy as cp 
import torch
import time
import jax.numpy as jnp  
import jax
import argparse

jax.config.update("jax_enable_x64", True)

def create_data(N,k):
    node_list = np.array([np.random.rand(N),
                          np.random.rand(N)]).reshape((N,2))
    link_list = []
    for i in range(N):
        distance = np.array([np.linalg.norm(node_list[i]-node_list[j]) 
                             for j in range(N)])
        neighbors = np.argsort(distance)[1:(k+1)]
        link = np.zeros((N,k))
        link[i,:] = -1
        link[neighbors,np.array(range(k))] = 1
        link2 = np.zeros((N,k))
        link2[i,:] = 1
        link2[neighbors,np.array(range(k))] = -1
        link_list.append(link)
        link_list.append(link2)
    A = np.hstack(link_list)
    A = np.unique(A,axis=1)
    p = np.random.permutation(A.shape[1])
    c = np.exp(np.random.rand(A.shape[1])*(np.log(5)-
                                           np.log(0.5))+np.log(0.5))
    return A[:,p], c

@jax.jit
def project(F,c):
    @jax.vmap 
    def update_array(arr, indices, values):
        return arr.at[indices].set(values)
    sort_index = jnp.argsort(-F.T)
    mat1 = -jnp.take_along_axis(-F.T, sort_index, axis=1)
    mat2 = (jnp.cumsum(mat1,axis=1)-c)/(jnp.arange(mat1.shape[1])+1)
    mat3_1 = jnp.where(mat1-mat2>0,mat1-mat2,jnp.inf)
    mat3_1ind = jnp.expand_dims(jnp.argmin(mat3_1,1),-1)
    mat3 = jnp.take_along_axis(mat2, mat3_1ind, axis=1)
    mat4 = jnp.where(mat1-mat3>0,mat1-mat3,0)
    F_project = update_array(jnp.zeros_like(mat4),sort_index,mat4).T
    F_plus = jnp.maximum(F,0)
    return jnp.where(F_plus.sum(axis=0)<=c[:,0],F_plus,F_project)

@jax.jit
def prox_util(Y, beta_weight):
    n1 = (Y - (Y**2 + 4*beta_weight)**0.5)/2
    n1 = jnp.fill_diagonal(n1,0,inplace=False)
    return n1

@jax.jit
def eval_obj(F,c,weight):
    f1 = (F>=-1e-4).all()
    f2 = (F.sum(axis=0)<=c+1e-4).all()
    f3 = XAt(jnp.zeros((F.shape[0],F.shape[0])),-F)
    f3 = jnp.fill_diagonal(f3,1,inplace=False)
    f4 = (f3>0).all()
    return jax.lax.cond((f1&f2)&f4, 
                        lambda x: (-weight*jnp.log(x)).sum(), 
                        lambda x: jnp.inf, f3)

@jax.jit
def compute_r(G,weight,pre_proj):
    minusFAt = XAt(jnp.zeros((G.shape[0],G.shape[0])),-G)
    minusFAt = jnp.fill_diagonal(minusFAt,1,inplace=False)
    inv_minusFAt = (1/minusFAt)*weight 
    inv_minusFAt = jnp.fill_diagonal(inv_minusFAt,0,inplace=False)
    nabla_u = XA(inv_minusFAt)
    v = (nabla_u**2).sum()
    H = G-pre_proj
    s = (H**2).sum()
    p = (H*nabla_u).sum() 
    r = jax.lax.cond((p>=0)&(s>0), 
                        lambda x: x[0]-x[1]**2/x[2], 
                        lambda x: x[0], [v,p,s])
    return jax.lax.cond((minusFAt>0).all(),
                        lambda x: r/(G.shape[0]*G.shape[1]),
                        lambda x: jnp.inf, r)

@jax.jit
def weight_update(F,Y,F_Y_0,pweight,eps_zero,eta):
    del_F = ((F-F_Y_0[0])**2).sum()**0.5
    del_Y = ((Y-F_Y_0[1])**2).sum()**0.5
    pweight = jax.lax.cond((del_F>eps_zero)&(del_Y>eps_zero), 
                        lambda x: (jnp.exp(0.5*jnp.log(x[1]/x[0])+
                                           0.5*jnp.log(x[2]))), 
                        lambda x: pweight, [del_F,del_Y,pweight])
    return eta/pweight, eta*pweight, pweight

@jax.jit
def XA(X):
    return jnp.take_along_axis(X,pos_ind,axis=1) \
            - jnp.take_along_axis(X,neg_ind,axis=1)

@jax.jit
@jax.vmap
def XAt(base, X):
    pos = base.at[pos_ind].add(X)
    res = pos.at[neg_ind].add(-X)
    return res

@jax.jit
def update(Y,F_half,alpha,beta,weight,c_exp,overrelax_rho):
    alpha_YA = alpha * XA(Y) 
    F_half_hat = project(F_half + alpha_YA, c_exp)
    Y_hat = prox_util(Y - beta*XAt(jnp.zeros_like(Y),2 * F_half_hat - F_half), beta * weight)
    F_half = overrelax_rho * F_half_hat + (1-overrelax_rho) * F_half
    Y = overrelax_rho * Y_hat + (1-overrelax_rho)*Y 
    return F_half_hat, alpha_YA, F_half, Y

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--q', type=int)
    parser.add_argument('--wu_it', type=int, default=100, required=False)
    parser.add_argument('--seed', type=int, default=0, required=False)
    parser.add_argument('--max_iter', type=int, default=np.inf, required=False)
    parser.add_argument('--eps', type=float, default=1e-2, required=False)
    parser.add_argument('--float64', action='store_true')
    parser.add_argument('--mosek_check', action='store_true')
    args = parser.parse_args()

    # create data
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n = args.n; q = args.q
    A, c = create_data(n,q) 
    weight = np.exp(np.random.rand(n,n)*(np.log(3)-np.log(0.3))+np.log(0.3))
    m = A.shape[1]
    print(f'{n=},{q=},{m=}')

    # sanity check with mosek 
    if args.mosek_check:
        print(f'START MOSEK SOLVE')
        F = cp.Variable(A.shape) 
        bimask = np.ones((n,n))
        np.fill_diagonal(bimask, 0)
        obj = -cp.sum(cp.multiply(weight,cp.log(-cp.multiply(bimask,F@A.T)+np.eye(n))))
        prob = cp.Problem(cp.Minimize(obj),[F>=0,F.T@np.ones(n)<=c])
        start_time = time.time()
        prob.solve(solver=cp.MOSEK)
        cvx_optimal = prob.value
        mosek_time = prob._solve_time
        cvx_F = F.value
        print('mosek time:', prob._solve_time)

    # PDHG algorithm 
    print(f'START PDHG SOLVE')
    if args.float64:
        print('using float64')
        A = jnp.array(A,dtype=jnp.float64)
        c = jnp.array(c,dtype=jnp.float64)
        weight = jnp.array(weight,dtype=jnp.float64)
    else:
        A = jnp.array(A,dtype=jnp.float32)
        c = jnp.array(c,dtype=jnp.float32)
        weight = jnp.array(weight,dtype=jnp.float32)
    pos_ind = jnp.where(A.T==1)[1].reshape(1,m) # index A matrix
    neg_ind = jnp.where(A.T==-1)[1].reshape(1,m) # index A matrix
    del A
    c_exp = jnp.expand_dims(c,-1)

    jax.device_put(0.).block_until_ready(); start_time = time.time() # start timing
    
    if args.float64:
        F_half = jnp.zeros((n,m),dtype=jnp.float64)
        Y = -jnp.ones((n,n),dtype=jnp.float64)
    else:
        F_half = jnp.zeros((n,m),dtype=jnp.float32)
        Y = -jnp.ones((n,n),dtype=jnp.float32)
    Y = jnp.fill_diagonal(Y,0,inplace=False)
    
    count = jnp.array([jnp.where(pos_ind==i)[0].shape[0] + \
            jnp.where(neg_ind==i)[0].shape[0] for i in range(n)])
    d_max = jnp.max(count).item()
    eta = 1/(2*d_max)**0.5
    if args.float64:
        pweight = 1.
        overrelax_rho = 1.9
        eps_zero = 1e-5
    else:
        pweight = np.float32(1.)
        overrelax_rho = np.float32(1.9)
        eps_zero = np.float32(1e-5)
    F_Y_0 = [F_half,Y]
    alpha = eta/pweight
    beta = eta*pweight
    wu_it = args.wu_it

    MAX_ITER = args.max_iter
    it = 0 

    while it < MAX_ITER:
        it += 1
        F_prev = F_half.clone()
        # PDHG update
        F_half_hat, alpha_YA, F_half, Y = update(Y,F_half,alpha,
                                                beta,weight,c_exp,overrelax_rho)
        # check stopping criterion
        if it%10 == 0:
            r = compute_r(F_half_hat,weight,F_prev+alpha_YA)
            residual = r.item()/(n*(n-1))
            print(f'{it=},{residual=}')
            if r/(n*(n-1))<args.eps:
                break
        # update primal weight
        if it%wu_it == 0:
            alpha, beta, pweight = weight_update(F_half,Y,F_Y_0,
                                                 pweight,eps_zero,eta)
            F_Y_0 = [F_half,Y]

    jax.device_put(0.).block_until_ready()
    print('pdmcf time:', time.time()-start_time)
    if args.mosek_check:
        # check normalized objective gap to MOSEK sol
        obj = eval_obj(F_half_hat,c,weight)
        pdmcf_mosek_diff = (obj-cvx_optimal).item()/(n*(n-1))
        normalized_objective = cvx_optimal.item()/(n*(n-1))
        print('normalized_objective:', normalized_objective)
        print('pdmcf_mosek_diff:', pdmcf_mosek_diff)
        













