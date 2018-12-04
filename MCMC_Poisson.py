import numpy as np
from scipy.stats import gamma
import math
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
    # 次のlambdaをご提案いたします（分散1の正規分布）
    lamb_next = np.random.normal(lamb, 0.5)

    # 尤度*事前分布の計算
    alpha_prior = 2
    beta_prior = 2
    L = np.exp(logLikelihood(lamb, data)) + gamma.pdf(lamb, alpha_prior, beta_prior)
    L_next = np.exp(logLikelihood(lamb_next, data)) + gamma.pdf(lamb_next, alpha_prior, beta_prior)

    # lambの更新
    if L <= L_next:
        lamb = lamb_next
    else:
        u = np.random.uniform(0,1)
        if L_next > u * L:
            lamb = lamb_next
    return lamb

if __name__ == "__main__":
    # まずはデータを得る。コードの都合上、lambdaは明記されてしまっているが、poisson(10,15)の10が推定したいパラメータ。とりあえず15個データをとる。
    data = np.random.poisson(10,25)
    print(data)

    # lambdaの初期値
    lamb = np.random.uniform(1, 100)

    MC = []
    MC.append(lamb)
    # 何回遷移させればいいか全くわからんからとりあえずいっぱい
    for i in range(10 ** 4):
        lamb = MHStep(lamb,data)
        MC.append(lamb)
    
    # 解析的に求めた事後分布との比較
    beta = 2+len(data)
    x = np.linspace(0,50,1000)
    y = gamma.pdf(x, 2+np.sum(data), scale=1./beta)
    plt.plot(x,y,'r-',lw=2)
    plt.hist(MC, density=True, bins=100)
    plt.savefig('fig.png')
    print(np.sum(data))