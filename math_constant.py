import math as mth
def constant(x):
    if x == 'Pi':
        return mth.pi
    elif x == 'Euler':
        return mth.e
    elif x == 'Phi':
        return (1 + mth.sqrt(5)) / 2
    elif x == 'Champernowne':
        A = '0.'
        B = 1
        for _ in range(100):
            A += str(B)
            B += 1
        return float(A)
    elif x == 'Tau':
        return 2 * mth.pi
    elif x == 'Apery':
        return 1.20205690315959428539973816151144999076498629234049
    elif x == 'Euler-Mascheroni':
        return 0.57721566490153286060651209008240243104215933593992
    elif x == 'Pythagoras':
        return mth.sqrt(2)
