""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/


% This main file sets up the steps of fitting a linear regression with an
% independent normal prior, centered at zero, on the weight parametes.  The 
% hyper parameters of the model are then beta, the precision of the noise
% , and alpha, the precisionn on the weights.  The problem is to find
% maximum likelihood estimators of alpha and beta with the weight
% parameters integrated out.  Variables of interested are listed here :
%
% d : covariate dimension of data
% X : design (covariate) matrix, with intercept column included
% Y : vector of observations
% w : vector of true weigts used for simulation of data
%
% alpha : weight precision (scalar)
% beta  : observation noise precision (scalar)
% m     : mean of weight distribtion in e_step (16 x 1)
% s     : covariance of weight distribution in e_step (16 x 16)
% ll    : approximate log likelihood of the joint distribution"""

import numpy as np
import scipy.io
import scipy.stats
from log_likelihood_linear_regression import log_likelihood_linear_regression
from e_step_linear_regression import e_step_linear_regression
from m_step_linear_regression import m_step_linear_regression

d = 15;
e = 0.00001;

data = scipy.io.loadmat("data_linear_regression.mat")
X = data["X"]
Y = data["Y"]

# % randomly set alpha, beta, and m to start
alpha = float(scipy.stats.dirichlet.rvs([1]))
beta = float(scipy.stats.dirichlet.rvs([1]))
m = scipy.stats.uniform.rvs(size=d + 1) * np.random.choice([-1, 1],
                                                           size=d + 1,
                                                           replace=True)

# % iterate until convergence
ll = log_likelihood_linear_regression(X, Y, m, beta)
print('log likelihood = %f' % ll)
print('alpha = %f' % alpha) 
print('beta = %f' % beta)
while (True):
    [m, s] = e_step_linear_regression(X, Y, alpha, beta)
    [alpha, beta] = m_step_linear_regression(X, Y, m, s)
    
    if (ll + e >= log_likelihood_linear_regression(X, Y, m, beta)): 
        break
    
    ll = log_likelihood_linear_regression(X, Y, m, beta)
    print('log likelihood = %f' % ll)
    print('alpha = %f' % alpha) 
    print('beta = %f' % beta)
    
