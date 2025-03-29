import numpy as np
np.random.seed(41)
X=lambda:np.random.randint(0,2) #Defining Random variable X which is 1 when Heads and 0 when Tails

def AveragePayoutmGames(m):
    TotalPayout=0.0
    for i in range(m):
        j=1
        while(X()==1):
            j+=1
        TotalPayout+=2**j
    AveragePayout=TotalPayout/m
    return AveragePayout
m1=100
m2=10000
m3=1000000
print(f"{AveragePayoutmGames(m1):.3f} {AveragePayoutmGames(m2):.3f} {AveragePayoutmGames(m3):.3f}")

