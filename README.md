## Quasi-Adam: Boosting exponential moving average performance with a 2-rank quasi-Newton Hessian approximation

### Usage

To run the operation on a 2-dim rosenbrock function use the command
```
    python rosenbrock.py --save-dir <dir> --iters <num iters>
```

where ```<dir>``` is the directory the file is to be saved in,
```<num iters>``` is the number of iterations the optimizer needs to be run.


## Result

In the following figure, you can see the results of the optimizer on the rosenbrock function

![The x-axis is the number of iterations, the y-axis is the value of y](https://github.com/aranganath/Adaptive-quasi-Newton-updates/blob/master/results.png)