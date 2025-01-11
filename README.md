# PDMCF: Solving Multicommodity Network Flow Problems on GPUs

This repo contains the source code for experiments for our PDMCF paper: XXXpaper linkXXX

## Repository Overview

We provide <code>pdmcf.py</code> for our torch implementation, <code>pdmcf_jax.py</code> for our jax implementation, and <code>pdmcf_warm.py</code> for reproducing our warm start results

## Quickstart
Clone the repo and run the following command
```
conda create -n pdmcf python=3.12
conda activate pdmcf
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirement.txt
conda install pytorch-scatter -c pyg
```
## PDMCF experiment
Run PDMCF method
```
python pdmcf.py --n 100 --q 10
```
where <code>--n</code> specifies number of nodes, <code>--q</code> specifies number of neighbors. One can also add <code>--mosek_check</code> to check with MOSEK result, note this requires to purchase MOSEK license. <code>--float64</code> can be added to switch from *float32* to *float64*, which gives more precise numerical result.
## PDMCF(JAX) experiment
Install JAX
```
pip install -U "jax[cuda12]"
```
Run PDMCF method
```
python pdmcf_jax.py --n 100 --q 10
```
similarly, <code>--mosek_check</code> can be added to check with MOSEK result, and <code>--float64</code> can be added to switch to higher accuracy.
## Warm start experiment
