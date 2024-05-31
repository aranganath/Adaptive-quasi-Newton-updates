import torch

from torch import optim
from utils.general_util import log
import argparse


parser = argparse.ArgumentParser(

    )

parser.add_argument('--iters', type=int, default=100)

args = parser.parse_args()

num_iters = args.iters

def main():
    func = lambda x: (1-x[0])**2 + (x[1] - x[0]**2)**2

    



    # fix the starting point
    x = torch.tensor([-1. , 1.], requires_grad=True)

    optimizer = optim.Adam(params=[x], lr=1e-3)

    for i in range(num_iters):
        y = func(x)
        log.infov("Training step {}: Loss: {}".format(i+1, y.item()))
        y.backward()
        optimizer.step()


if __name__=="__main__":
    main()