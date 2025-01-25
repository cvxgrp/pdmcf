import numpy as np  
import torch
import time
from torch_scatter import scatter, scatter_add
import argparse

def create_data(N,k):
    node_list = np.array([np.random.rand(N),np.random.rand(N)]).reshape((N,2))
    link_list = []
    for i in range(N):
        distance = np.array([np.linalg.norm(node_list[i]-node_list[j]) for j in range(N)])
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
    c = np.exp(np.random.rand(A.shape[1])*(np.log(5)-np.log(0.5))+np.log(0.5))
    return A[:,p], c

def project(F,c):
    sorted, ind = torch.sort(-F.T)
    mat1 = -sorted; del sorted
    mat2 = (torch.cumsum(mat1,dim=1)-c)/(torch.arange(mat1.shape[1]).to(F.device)+1)
    mat3_1 = torch.where(mat1-mat2>0,mat1-mat2,torch.inf)
    mat3_1ind = torch.min(mat3_1,1)[1].unsqueeze(-1); del mat3_1
    mat3 = torch.gather(mat2,1,mat3_1ind); del mat2
    mat4 = mat3.expand(F.shape[1],F.shape[0])
    mat5 = torch.where(mat1-mat4>0,mat1-mat4,0)
    F_project = scatter(mat5,ind,1).T
    F_plus = torch.maximum(F,torch.zeros_like(F))
    col_ind = torch.where(F_plus.sum(dim=0)<=c[:,0])[0]
    F_project[:,col_ind] = F_plus[:,col_ind]
    return F_project

def prox_util(Y,beta_weight):
    n1 = (Y - (Y**2 + 4 * beta_weight)**0.5)/2
    n1.fill_diagonal_(0)
    return n1

def eval_obj(F,pos_ind,neg_ind,c,weight):
    f1 = (F>=-1e-4).all()
    f2 = (F.sum(dim=0)<=c+1e-4).all()
    f3 = scatter_add(F,neg_ind,1)-scatter_add(F,pos_ind,1)
    f3.fill_diagonal_(1)
    f4 = (f3>0).all()
    if not (f1 and f2 and f4):
        return torch.inf 
    return ((-weight*torch.log(f3)).sum()).item()

def compute_r(F,pre_proj,neg_ind,pos_ind,weight):
    minusFAt = scatter_add(F,neg_ind,1) - scatter_add(F,pos_ind,1) 
    minusFAt.fill_diagonal_(1)
    if not (minusFAt>0).all():
        return torch.Tensor([torch.inf])
    inv_minusFAt = (1/minusFAt)*weight 
    inv_minusFAt.fill_diagonal_(0)
    nabla_u = torch.gather(inv_minusFAt,1,pos_ind.expand(F.shape))-\
                torch.gather(inv_minusFAt,1,neg_ind.expand(F.shape))
    v = (nabla_u**2).sum()
    s = ((F-pre_proj)**2).sum()
    p = ((F-pre_proj)*nabla_u).sum() 
    r = v-p**2/s if (p>=0 and s>0) else v
    r = r/(F.shape[0]*F.shape[1])
    return r

def weight_update(F,Y,pweight,eps_zero,eta,F_init,Y_init):
    del_F = ((F-F_init)**2).sum()**0.5
    del_Y = ((Y-Y_init)**2).sum()**0.5
    if del_F>eps_zero and del_Y>eps_zero:
        pweight = torch.exp(0.5*torch.log(del_Y/del_F)+\
            0.5*torch.log(torch.Tensor([pweight])).item())
    return eta/pweight, eta*pweight, pweight

def perturb(weight,ratio):
    ret = weight + (2*(torch.rand((weight.shape))<0.5)-1).to(device)*(weight*ratio)
    return ret

if __name__ == "__main__":
    device = 'cuda:0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--q', type=int)
    parser.add_argument('--wu_it', type=int, default=100, required=False)
    parser.add_argument('--nu', type=float, default=0.1, required=False)
    parser.add_argument('--seed', type=int, default=0, required=False)
    parser.add_argument('--max_iter', type=int, default=np.inf, required=False)
    parser.add_argument('--float64', action='store_true')
    args = parser.parse_args()

    # create data
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n = args.n; q = args.q
    A, c = create_data(n,q) 
    weight = np.exp(np.random.rand(n,n)*(np.log(3)-np.log(0.3))+np.log(0.3))
    m = A.shape[1]
    print(f'{n=},{q=},{m=}')


    # PDHG algorithm 
    print(f'START PDHG SOLVE')
    A = torch.Tensor(A)
    pos_ind = torch.where(torch.Tensor(A).T==1)[1].to(device)
    neg_ind = torch.where(torch.Tensor(A).T==-1)[1].to(device)
    del A
    c = torch.Tensor(c).to(device)
    weight = torch.Tensor(weight).to(device)
    c_exp = c.expand(n, m).T
    
    F_half = torch.zeros((n,m)).to(device)
    Y = -torch.ones((n,n)).to(device)
    Y.fill_diagonal_(0)
    if args.float64:
        print('using float64')
        A = A.double()
        c = c.double()
        c_exp = c.expand(n, m).T
        weight = weight.double()
        F_half = F_half.double()
        Y = Y.double()

    count = torch.Tensor([torch.where(pos_ind==i)[0].shape[0] + \
            torch.where(neg_ind==i)[0].shape[0] for i in range(n)])
    d_max = torch.max(count)
    eta = 1/(2*d_max)**0.5
    pweight = 1
    eps_zero = 1e-5
    F_Y_0 = [F_half,Y]
    alpha = eta/pweight 
    beta = eta*pweight
    overrelax_rho = 1.9
    wu_it = args.wu_it
    weight_original = weight.clone()
    # perturb log weight a
    weight = perturb(weight_original,args.nu)
    print(f'weight perturb ratio: {args.nu}')

    MAX_ITER = args.max_iter
    it = 0

    # warm up solve
    while it < MAX_ITER:
        alpha_YA = torch.gather(alpha * Y,1,pos_ind.expand(n, m))-\
                torch.gather(alpha * Y,1,neg_ind.expand(n, m))
        F_prev = F_half.clone()
        F_half_hat = project(F_half + alpha_YA, c_exp)
        F_new = 2*F_half_hat - F_half
        F_At = scatter_add(beta * F_new,pos_ind,1)-scatter_add(beta * F_new,neg_ind,1)
        Y_hat = prox_util(Y - F_At, beta * weight)
        F_half = overrelax_rho*F_half_hat + (1-overrelax_rho)*F_half 
        Y = overrelax_rho * Y_hat + (1-overrelax_rho)*Y 
        it += 1
        if it%10 == 0:
            r = compute_r(F_half_hat,F_prev + alpha_YA,neg_ind,pos_ind,weight)
            residual = r.item()/(n*(n-1))
            print(f'warm up solve: {it=},{residual=}')
            if r/(n*(n-1))<torch.inf:
                break
        if it%wu_it == 0:
            alpha, beta, pweight = weight_update(F_half,Y,pweight,
                                                 eps_zero,eta,F_Y_0[0],F_Y_0[1])
            F_Y_0 = [F_half,Y]

    # true solve
    it = 0
    weight = weight_original.clone()
    torch.cuda.synchronize()
    start_time = time.time() # start timing
    F_Y_0 = [F_half,Y]
    while it < MAX_ITER:
        alpha_YA = torch.gather(alpha * Y,1,pos_ind.expand(n, m))-\
                torch.gather(alpha * Y,1,neg_ind.expand(n, m))
        F_prev = F_half.clone()
        F_half_hat = project(F_half + alpha_YA, c_exp)
        F_new = 2*F_half_hat-F_half
        F_At = scatter_add(beta * F_new,pos_ind,1)-scatter_add(beta * F_new,neg_ind,1)
        Y_hat = prox_util(Y-F_At, beta * weight)
        F_half = overrelax_rho*F_half_hat + (1-overrelax_rho)*F_half 
        Y = overrelax_rho * Y_hat + (1-overrelax_rho)*Y 
        it += 1
        if it%10 == 0:
            r = compute_r(F_half_hat,F_prev+alpha_YA,neg_ind,pos_ind,weight)
            residual = r.item()/(n*(n-1))
            print(f'true solve: {it=},{residual=}')
            if r/(n*(n-1))<1e-2:
                break
        if it%wu_it == 0:
            alpha, beta, pweight = weight_update(F_half,Y,pweight,
                                                 eps_zero,eta,F_Y_0[0],F_Y_0[1])
            F_Y_0 = [F_half,Y]

    torch.cuda.synchronize()
    print('pdmcf (with warm start) time:', time.time()-start_time)



