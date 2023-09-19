import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import random
import math



def load_csv(Folder_Path):
    convergence=np.loadtxt(f'{Folder_Path}/convergence.csv')
    constrained_violation_curve=np.loadtxt(f'{Folder_Path}/constrained_violation_curve.csv')
    bestIndividual=np.loadtxt(f'{Folder_Path}/bestIndividual.csv')
    return convergence,constrained_violation_curve,bestIndividual



# def plotBox(Alg_bestIndividual):
    
    
        
#     reshape_Alg =np.reshape(Alg_bestIndividual,(2,base_numbers,ue_numbers))
#     check_conectional = (reshape_Alg[0] >= 500)
#     rate_table=calculator_throughtput(reshape_Alg,check_conectional)
#     every_ue_rate=np.sum(rate_table,axis=0)
#     return every_ue_rate


if __name__ == "__main__":
    
    np.random.seed(000)
    random.seed(000)

    # Alg_list =  ['GA','HHO','HHO_SMA','HHO_SMA_Chaos','HHO_SMA_DE','OriginalSMA']
    # Alg_list =  ['HHO','OriginalSMA']
    Alg_list=['HHO']
    test_function = ['f3']
    # test_function = ['f1','f2','f3']
    # Alg_list =  ['HHO_DE_Chaos']
    Init_Path = './2023-06-14'
    StartTime = 0
    EndTime = 30
    fig_EE,ax_EE = plt.subplots()
    fig_violation,ax_violation = plt.subplots()
    fig_box, ax_box = plt.subplots()
    box_list=[]
    boxname_list=[]
    std_EE = []
    std_violation = []
    std_curve = []
    avg_curve =  []
    med_curve = []
    for index,Alg in enumerate(Alg_list):
        EE_Alg = []
        violation_Alg = []
        for test in test_function:
            fig_curvex, ax_curvex = plt.subplots()
            for Time in range(StartTime,EndTime):
                if Time == 0:
                    avg_convergence,avg_constrained_violation_curve,bestIndividual=\
                    load_csv(f'{Init_Path}/{Alg}/{test}/{Time}')
                    # avg_throughput=plotBox(bestIndividual)
                    EE_Alg.append(avg_convergence[-1])
                    violation_Alg.append(avg_constrained_violation_curve[-1])
                else:
                    convergence,constrained_violation_curve,bestIndividual=\
                    load_csv(f'{Init_Path}/{Alg}/{test}/{Time}')
                    EE_Alg.append(convergence[-1])
                    violation_Alg.append(constrained_violation_curve[-1])
                    avg_convergence = (avg_convergence + convergence)/2
                    avg_constrained_violation_curve = (avg_constrained_violation_curve + constrained_violation_curve) /2
                    # avg_bestIndividual = (avg_bestIndividual + bestIndividual) /2
                    # avg_throughput=(plotBox(bestIndividual)+avg_throughput)/2
            # std_EE.append(np.log1p(EE_Alg))
            
            ax_curvex.plot(avg_convergence[200:400],label=f'{Alg}_{test}')
            ax_curvex.set_title(f'{Alg} test {test} ')
            ax_curvex.set_xlabel("iteration")
            ax_curvex.set_ylabel("score")
            ax_curvex.set_yscale("log",basey=10)
            if Alg=='HHO' and test=='f3':
                # y_ticks=[0,10E-10,10E-20,10E-30]
                # ax_curvex.set_yticks(y_ticks)
                x_ticks=[50,100,150,200]
                ax_curvex.set_xticks(x_ticks)
            fig_curvex.savefig(f'{Init_Path}/{Alg}_{test}.jpg')
            std_curve.append(np.std(np.array(EE_Alg)))
            print(f'{Alg}_{test}_std:{np.std(np.array(EE_Alg))}')
            avg_curve.append(np.mean(np.array(EE_Alg)))
            print(f'{Alg}_{test}_avg:{np.mean(np.array(EE_Alg))}')
            med_curve.append(np.median(np.array(EE_Alg)))
            print(f'{Alg}_{test}_median:{np.median(np.array(EE_Alg))}')
            std_violation.append(np.std(np.array(violation_Alg)))    
            # throughput=plotBox(avg_bestIndividual)
            
            # y = avg_throughput
            # x = np.random.normal(index + 1, 0.04, size=len(y))
            # ax_box.scatter(x, y, alpha=0.5, s=8, color='black')
            # box_list.append(y)
            # boxname_list.append(f'{Alg}')
            ax_EE.plot(avg_convergence,label=f'{Alg}_{test}')
            ax_violation.plot(avg_constrained_violation_curve,label=f'{Alg}_{test}')
        
    # ax_EE.set_yscale("log",basey=10)
    fig_EE.legend()
    fig_violation.legend()
    ax_EE.set_title("test function of f2")
    ax_EE.set_xlabel("iteration")
    # ax_EE.set_xlabel("iteration")
    fig_EE.savefig(f'{Init_Path}/EE.jpg')
    # fig_violation.savefig(f'{Init_Path}/violation.jpg')
    # save throughput csv
    csv_data = DataFrame(std_EE)
    csv_data =csv_data.T
    csv_data.rename(columns={index:name for index,name in enumerate(boxname_list)},inplace=True)
    csv_data.to_csv(f'{Init_Path}/EE_std.csv')
    csv_data = DataFrame(std_violation)
    csv_data =csv_data.T
    csv_data.rename(columns={index:name for index,name in enumerate(boxname_list)},inplace=True)
    csv_data.to_csv(f'{Init_Path}/std_violation.csv')
    print(f'std:{std_curve}')
    print(f'avg:{avg_curve}')
    print(f'med:{med_curve}')