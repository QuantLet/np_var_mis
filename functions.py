import numpy as np
import random
from scipy.stats import norm, bernoulli



# Nonparametric regression 

def eps(mu,sigma,n):
    random.seed(24)
    return np.random.normal(mu,sigma,n)

# Variance function
def sigma(t):
    #return 0.5*(1.5-t) Efromovich
    #return 0.4 + 2*t #https://www.ruhr-uni-bochum.de/imperia/md/content/mathematik3/publications/proctest21.pdf
    #return np.exp(-0.25*t) #Dette 
    return 0.4 * np.exp(-2*(t**2)) + 0.2


def sigma_reg(t,n):
    #return 0.5*(1.5-t) Efromovich
    #return 0.4 + 2*t #https://www.ruhr-uni-bochum.de/imperia/md/content/mathematik3/publications/proctest21.pdf
    return 0.4 * np.exp(-2*(t**2)) + 0.2 + eps(0, 1, n)

# True unknown relationship
def f(X,n):
    return (np.sin(2*np.pi*(X)**3))**3 + sigma(X)*eps(0, 1,  n)

def fan(X,n,a):
    return a*(X+2*np.exp(-16* (X**2))) + sigma(X)*eps(0, 1,  n)


# Nadaraya Watson Estimator with a Gaussian Kernel 
def nw(h,x,X,y):
    num = sum(y*norm.pdf((x-X)/h))
    dem = sum(norm.pdf((x-X)/h))
    return num/dem

#### Missing values ß

#Logistic probability of observation 
def pi(y,b0,b1):
    lin = b0 + b1 *y 
    return 1/(1+np.exp(-lin))

# NW-estimated pi_hat 
def pi_hat(h,y_i,Y,p,n):
    omega = bernoulli.rvs(p, size=n)
    num = sum(omega*norm.pdf((y_i-Y)/h))
    dem = sum(norm.pdf((y_i-Y)/h))
    return num/dem

# HW-type NW estimator 
def nw_mis(h,x,X,y,p,n):
    omega = bernoulli.rvs(p, size=n)
    y = (y*omega)/p
    num = sum(y*norm.pdf((x-X)/h))
    dem = sum((omega/p)*norm.pdf((x-X)/h))
    return num/dem

def nw_cc(h,x,X,y,p,n):
    omega = bernoulli.rvs(p, size=n)
    y = (y*omega)
    num = sum(y*norm.pdf((x-X)/h))
    dem = sum(omega*norm.pdf((x-X)/h))
    return num/dem


