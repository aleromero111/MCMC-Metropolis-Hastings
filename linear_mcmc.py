import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from scipy.optimize import minimize

from getdist import loadMCSamples, plots

# INPUTS #

x = np.array([
    0.8308,
    0.5853,
    0.5497,
    0.9172,
    0.2858,
    0.7572,
    0.7537,
    0.3804,
    0.5678,
    0.0759
])

y = np.array([
    0.9160,
    0.7958,
    0.8219,
    1.3757,
    0.4191,
    0.9759,
    0.9455,
    0.3871,
    0.7239,
    0.0964
])

sigma=0.1
prior_limits=[-10,10]
nsteps=10000 

bounds = [(-10,10), (-10,10), (-10,2)] #limits allowed to explore in the parameter space
step_size = np.array([0.05, 0.05, 0.01]) 
burnin = int(0.2*nsteps) #Burn in to eliminate the first steps before the convergence

# FUNCTIONS #

def linear_model(theta, x):
    return theta[0] + theta[1] * x


#log likelihood, coge una gaussiana centrada en el punto mu predicho, mu=a+b*x+lambda*N(0,sigma), y lo compara con el dato "y" que tenemos
#Si el dato cae cerca de la prediccion, Log L es grande y si cae lejos, pequeño
def log_likelihood(th,x,y,model):
    logsigma = th[-1]
    sigma = np.exp(logsigma) #Using log sigma to avoid negative values, doesn't change the results
    ymod=model(th[:-1],x) #with the parameters th, this is the value that y should have
    if sigma<=1e-10:
        return -np.inf

    return np.sum(stats.norm.logpdf(y,loc=ymod,scale=sigma)) #then, the function calculates the deviation from the expected value via max likelihood

def log_prior(theta, prior_limits):

    if np.all(theta[:-1] > prior_limits[0]) and np.all(theta[:-1] < prior_limits[1]) and -10 < theta[-1] < 2:
        return 0.0

    return -np.inf

def log_posterior(theta,prior_limits,x,y,model):

    lp = log_prior(theta,prior_limits)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta,x,y,model)

# MCMC function

def metropolis(log_posterior,theta0,nsteps,step_size,prior_limits,x,y,model):
    ndim=len(theta0)
    chain = np.zeros((nsteps,ndim))
    theta = theta0.copy()
    accepted=0
    logp=log_posterior(theta0,prior_limits,x,y,model)

    for i in range(nsteps):
        proposal = theta + step_size*np.random.randn(ndim)
        logp_new=log_posterior(proposal,prior_limits,x,y,model)

        if np.log(np.random.rand()) < logp_new - logp: # Metropolis-Hastings acceptance criterion
            theta=proposal
            logp=logp_new
            accepted += 1

        chain[i]=theta
    
    print("acceptance rate:", accepted/nsteps)

    return chain

# We calculate the initial parameters' estimators to introduce them in the MCMC
results = minimize(lambda th: -log_likelihood(th,x,y,linear_model),x0=[0,1,np.log(0.1)],bounds=bounds,method='L-BFGS-B')
theta0=np.zeros(len(results.x))

#The chain starts running from the MLE with a little perturbation, to allow the algorithm to converge faster
theta0 = results.x.copy() + 1e-4*np.random.randn(len(results.x)) 

# CHAIN #
chain = metropolis(log_posterior,
    theta0,
    nsteps,
    step_size,
    prior_limits,
    x=x,
    y=y,
    model=linear_model
)

samples = chain[burnin:] #Removing the first steps 
sigma_samples = np.exp(samples[:, -1]) 

theta_mean = np.mean(samples, axis=0)  
theta_mean[-1] = np.mean(sigma_samples) 
theta_median=np.median(sigma_samples)

# RESULTS AND PLOTS #

print(theta_mean,theta_median)

plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(chain[:,0])
plt.ylabel(r"$\theta_0$")

plt.subplot(3,1,2)
plt.plot(chain[:,1])
plt.ylabel(r"$\theta_1$")

plt.subplot(3,1,3)
plt.plot(np.exp(chain[:,-1]))
plt.ylabel(r"$\sigma$")

plt.xlabel("step")

plt.tight_layout()

#Plot the linear model and data

xreg=np.linspace(min(x)-0.1,max(x)+0.1, 20)
yreg=theta_mean[0]+xreg*theta_mean[1]

plt.figure(figsize=(10,6))
plt.errorbar(x,y,theta_mean[2],fmt='o',capsize=4)
plt.plot(xreg,yreg)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Linear regression and data')

#Covariance ellipse

Cov = np.cov(samples.T)
errors = np.sqrt(np.diag(Cov))

cov_inv = np.linalg.inv(Cov[:-1,:-1])

t1 = np.linspace(theta_mean[0]-4*errors[0], theta_mean[0]+4*errors[0], 300)
t2 = np.linspace(theta_mean[1]-4*errors[1], theta_mean[1]+4*errors[1], 300)
T1, T2 = np.meshgrid(t1, t2)

d1 = T1 - theta_mean[0]
d2 = T2 - theta_mean[1]

Q = (cov_inv[0,0]*d1**2 +
     2*cov_inv[0,1]*d1*d2 +
     cov_inv[1,1]*d2**2)

#Plot the covariance ellipse for 1sigma, 2sigma, 3sigma

plt.figure(figsize=(10,6))
plt.contour(t1, t2, Q, levels=[2.30, 6.17, 11.8], colors=['blue','green','red'])
plt.scatter(samples[:,0],samples[:,1],s=2)
plt.scatter(theta_mean[0], theta_mean[1], color='red') # Centre
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.title(r'Covariance ellipses for $\sigma,2\sigma,3\sigma$')
plt.show()

