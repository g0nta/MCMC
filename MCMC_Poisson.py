import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

def logFractrial(x):
    result = 0.0
    for i in range(x):
        result += np.log(i+1)
    return result

def logLikelihood(lamb, data):
    # lambは常に0より大きいので、ここで-infinity返して後の更新処理で更新されないようにする
    if lamb <= 0:
        return -np.inf
    # calc log likelihood
    result = 0.0
    for x in data:
        result += (x * np.log(lamb)) - logFractrial(x)
    result -= len(data) * lamb
    return result

# MH法の1ステップ
def MHStep(lamb, data):
    # Propose a next lambda by Normal dist
    lamb_next = np.random.normal(lamb, 0.5)

    # Calc likelihood * prior dist
    alpha_prior = 2
    beta_prior = 2
    gamma_prior = gamma(a=alpha_prior, scale=1./beta_prior) # 事前分布
    L = np.exp(logLikelihood(lamb, data)) * gamma_prior.pdf(lamb)
    L_next = np.exp(logLikelihood(lamb_next, data)) * gamma_prior.pdf(lamb_next)

    # Update lamb
    if L <= L_next:
        lamb = lamb_next
    else:
        u = np.random.uniform(0,1)
        if L_next > u * L:
            lamb = lamb_next
    return lamb

if __name__ == "__main__":
    data = np.random.poisson(10,25)
    print(data)

    # Initialize lambda
    lamb = np.random.uniform(1, 100)

    hist = []
    hist.append(lamb)

    for i in range(10 ** 5):
        lamb = MHStep(lamb,data)
        hist.append(lamb)
    
    # 解析的に求めた事後分布との比較
    gamma_posterior = gamma(a=2+np.sum(data), scale=1./(2+len(data)))
    x = np.linspace(0,50,1000)
    y = gamma_posterior.pdf(x)
    plt.plot(x,y,'r-',lw=2)

    sample = hist[1000:]
    plt.hist(sample, density=True, bins=100)
    plt.savefig('fig.png')
    print(np.sum(data))
    print(np.average(sample))