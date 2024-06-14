import torch

# from torch import optim as toptim
from utils.general_util import log
from optim.adam import quasiAdam
import argparse
# import logging
import pickle as pkl


parser = argparse.ArgumentParser()

parser.add_argument('--iters', type=int, default=100)
parser.add_argument('--save-file', type=str, default="results.pkl")
args = parser.parse_args()

num_iters = args.iters

results = []
def main():
    func = lambda x: (1-x[0])**2 + (x[1] - x[0]**2)**2

    # fix the starting point
    x = torch.tensor([-1. , 1.], requires_grad=True)

    optimizer = quasiAdam(params=[x], lr=1e-2, betas=(0.5, 0.55))
    # optimizer = toptim.Adam(params=[x], lr=1e-2, betas=(0.1, 0.01))

    for i in range(num_iters):
        y = func(x)
        strx = str(x)
        log.infov("Training step {}: Iterate: {} Loss: {}".format(i+1, strx, y.item()))
        results.append(y.item())
        optimizer.zero_grad()
        y.backward()
        optimizer.step()

    
    with open(args.save_file, 'wb') as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()