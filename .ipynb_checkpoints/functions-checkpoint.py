import numpy as np
import random
from scipy.stats import norm, bernoulli



# Nonparametric regression 

def eps(mu,sigma,n):
    random.seed(240)
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
    return 0.4 * np.exp(-2*(t**2)) + 0.2 

# True unknown relationship
def f(X,n):
    return (np.sin(2*np.pi*(X)**3))**3 + sigma(X)*eps(0, 1,  n)

def fan(X,n,a,eps):
    return a*(X+2*np.exp(-16* (X**2))) + sigma(X)*eps


# Nadaraya Watson Estimator with a Gaussian Kernel 
def nw(h,x,X,y):
    num = sum(y*norm.pdf((x-X)/h))
    den = sum(norm.pdf((x-X)/h))
    return num/den

#### Missing values ß

#Logistic probability of observation 
def pi(y,b0,b1):
    lin = b0 + b1 *y 
    return 1/(1+np.exp(-lin))

# NW-estimated pi_hat 
def pi_hat(h,y_i,Y,p,omega):
    random.seed(24)
    num = sum(omega*norm.pdf((y_i-Y)/h))
    dem = sum(norm.pdf((y_i-Y)/h))
    return num/dem

# HW-type NW estimator 
def nw_mis(h,x,X,y,p,omega):
    y = (y*omega)/p
    num = sum(y*norm.pdf((x-X)/h))
    dem = sum((omega/p)*norm.pdf((x-X)/h))
    return num/dem

def nw_cc(h,x,X,y,p,omega):
    y = (y*omega)
    num = sum(y*norm.pdf((x-X)/h))
    dem = sum(omega*norm.pdf((x-X)/h))
    return num/dem

#### Residual based

# Nadaraya Watson Estimator with a Gaussian Kernel 
def sigma_res(h,x,X,r):
    num = sum(r*norm.pdf((x-X)/h))
    dem = sum(norm.pdf((x-X)/h))
    return np.sqrt(num/dem)


# HW-type NW estimator 
def sigma_mis_res(h,x,X,r,p,omega):
    r = (r*omega)/p
    num = sum(r*norm.pdf((x-X)/h))
    dem = sum((omega/p)*norm.pdf((x-X)/h))
    return np.sqrt(num/dem)

#### Difference-based estimator 

#We need a Kernel that sums up to one. Choose weights better

def diff_vol(h,x,X,y):
    diff = []
    for i in range(1,len(y)):
        diff.append(((y[i-1]-y[i])**2)/2)
    num = sum(diff*norm.pdf((x-X[:(len(X)-1)])/h))
    den = sum(norm.pdf((x-X[:(len(X)-1)])/h))
    return np.sqrt(num/den)



def diff_vol_mis(h,x,X,y,p,omega):
    diff = []
    for i in range(1,len(y)):
        diff.append(((y[i-1]-y[i])**2)/2)
    p = p[:(len(p)-1)]
    diff = diff/p
    num = sum(diff*norm.pdf((x-X[:(len(X)-1)])/h))
    den = sum(norm.pdf((x-X[:(len(X)-1)])/h)/p)
    return np.sqrt(num/den)

def r(h):
    #print(1)
    summe = 0 
    for j in range(len(df)):
        df1 = df.drop([j],axis=0)
        y_hat = nw_mis(h,df.x[j],df1.x,df1.y,pi(df1.y,b0,b1),df1.omega)
        summe = summe + (df.y[j]-y_hat)**2
    return summe/len(df)
