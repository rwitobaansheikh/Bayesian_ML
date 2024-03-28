#
# CM50268_CW2_setup
#
# Support code for Coursework 2 - Bayesian Sampling
#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, spatial


class DataGenerator:

    """Generate data for simple prediction modelling"""

    def __init__(self, m=3, r=1, noise=0.2, rand_offset=3):
        rs = rand_offset
        self._States = {'TRAIN': 0+rs, 'VALIDATION': 1+rs, 'TEST': 2+rs}
        #
        self.xmin = 0
        self.xmax = 10
        self._noise_std = noise
        self._M = m
        self._r = r
        #
        state = rs+1000  # Different state for generator
        wstd = 1
        self._Centres = np.linspace(0, 1, self._M)*self.xmax
        self._RBF = RBFGenerator(self._Centres, width=self._r)
        self._W = stats.norm.rvs(size=(self._M, 1), scale=wstd, random_state=state)
        #
        
    # Private
    # 
    def _make_data(self, name, N, noise_std=0.0):
        state = self._States[name]
        x = np.sort(stats.uniform.rvs(loc=0, scale=self.xmax, size=(N, 1),
                                      random_state=state), axis=0)
        PHI = self._RBF.evaluate(x)
        fy = PHI @ self._W
        if noise_std > 0:
            e = stats.norm.rvs(size=(N, 1), scale=noise_std, random_state=state)
            y = fy + e
            print("Measured noise std. dev. = {0:.3f}".format(np.std(e)))
        else:
            y = fy
        #
        return x, y
        
    # Public
    # 
    def get_data(self, name, N):
        name = name.upper()
        if name == 'TRAIN':
            return self._make_data(name, N, self._noise_std)
        elif name == 'VALIDATION':
            return self._make_data(name, N, self._noise_std)
        elif name == 'TEST':
            return self._make_data(name, N, 0)
        else:
            raise ValueError('Invalid data set name')
        

class RBFGenerator:

    """Generate Gausian RBF basis matrices"""

    def __init__(self, Centres, width=1):
        self._r = width
        self._M = len(Centres)
        self._Cent = Centres.reshape((self._M, 1))

    def evaluate(self, X):
        PHI = np.exp(-spatial.distance.cdist(X, self._Cent, metric="sqeuclidean") /
                     (self._r**2))
        #
        return PHI
    

# Specimen functions as required for Lab 1 previously
#
# Weight posterior
#
def compute_posterior(PHI, y, alph, s2):
    
    #####
    ##### Paste Your implementation from Part 1 here
    #####
    Mu=np.linalg.inv(PHI.T @ PHI + s2*alph*np.identity(PHI.shape[1]))@PHI.T@y
    SIGMA=np.linalg.inv(PHI.T @ PHI + s2*alph*np.identity(PHI.shape[1]))*s2
    return Mu, SIGMA


# Marginal log likelihood
#
# Version 1 Log Marginal (ideal)
#
def compute_log_marginal(PHI, y, alph, s2):
    
    #####
    ##### Paste Your implementation from Part 1 here
    #####
    N, M = PHI.shape

    C = s2 * np.eye(N) + PHI @ PHI.T / alph

    logML = -N/2 * np.log(2*np.pi)

    _, log_det = np.linalg.slogdet(C)

    logML -= log_det / 2

    logML -= y.T @ np.linalg.inv(C) @ y / 2
    return logML



# Convenient plotting: optional
#
def plot_regression(x_train, y_train, x_test=None, y_test=None, y=None):
    #
    lw = 2.5
    if y_test is not None:
        plt.plot(x_test, y_test, '--', color='GoldenRod', lw=lw)
    #
    if y is not None:
        plt.plot(x_test, y, '-', color='FireBrick', lw=lw)
    #
    plt.plot(x_train, y_train, 'o', ms=8, mew=3, color='black',
             markeredgecolor=[1, 1, 1, 0.8])
    #
    plt.xlabel("$x$", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
