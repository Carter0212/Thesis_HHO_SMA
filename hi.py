from GA import geneticalgorithm
import numpy as np
def f(X):
    return np.sum(X)

if __name__ == '__main__':
    # varbound=np.array([[0,10]]*3)
    # GAmd1 = geneticalgorithm(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)
    # GAmd1.run()
    a=np.array([0,1,2,3,4,5,6,7,8,9,10])
    print(a[:5])





