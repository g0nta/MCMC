import numpy as np
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

    # 尤度の計算
    logL = logLikelihood(lamb, data)
    logL_next = logLikelihood(lamb_next, data)

    # lambの更新
    # 事前分布が一様分布なので尤度比較するだけでいい
    if logL <= logL_next:
        lamb = lamb_next
    else:
        u = np.random.uniform(0,1)
        if logL_next > np.log(u) + logL:
            lamb = lamb_next
    return lamb

if __name__ == "__main__":
    # 推定したいのはPoisson分布のパラメータlambda.
    # MCMCでlambdaを推定する。lambdaの事前分布はめんどくさいので一様分布とする。

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
    
    # 描画
    plt.plot(MC)
    plt.savefig('Marcov_Chain.png')

    # とりあえず最後の5000個の平均を推定値とする
    sample = list(np.array_split(MC, 2))[1]
    print(np.average(sample))
    