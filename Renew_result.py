import numpy as np
import matplotlib.pyplot as plt

def load_csv(Folder_Path):
    convergence=np.loadtxt(f'{Folder_Path}/convergence.csv')
    constrained_violation_curve=np.loadtxt(f'{Folder_Path}/constrained_violation_curve.csv')
    bestIndividual=np.loadtxt(f'{Folder_Path}/bestIndividual.csv')
    return convergence,constrained_violation_curve,bestIndividual


if __name__ == "__main__":
    Alg_list =  ['GA','HHO','HHO_SMA','HHO_SMA_Chaos','HHO_SMA_DE','OriginalSMA']
    Init_Path = './2023-05-28'
    StartTime = 0
    EndTime = 25
    fig_EE,ax_EE = plt.subplots()
    fig_violation,ax_violation = plt.subplots()
    for Alg in Alg_list:
        for Time in range(StartTime,EndTime):
            if Time == 0:
                avg_convergence,avg_constrained_violation_curve,avg_bestIndividual=\
                load_csv(f'{Init_Path}/{Alg}/{Time}')
            else:
                convergence,constrained_violation_curve,bestIndividual=\
                load_csv(f'{Init_Path}/{Alg}/{Time}')
                avg_convergence = (avg_convergence + convergence)/2
                avg_constrained_violation_curve = (avg_constrained_violation_curve + constrained_violation_curve) /2
                avg_bestIndividual = (avg_bestIndividual + bestIndividual) /2

        ax_EE.plot(avg_convergence,label=f'{Alg}')
        ax_violation.plot(avg_constrained_violation_curve,label=f'{Alg}')
    fig_violation.legend()
    fig_EE.savefig(f'{Init_Path}/EE.jpg')
    fig_violation.savefig(f'{Init_Path}/violation.jpg')
        