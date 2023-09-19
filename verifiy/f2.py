
import numpy as np
from Alg import HHO,BaseSMA,OriginalSMA,HHO_SMA,GA,HHO_SMA_ch,HHO_SMA_Chaos,HHO_SMA_DE,HHO_DE_Chaos
import math
import Alg
import random
import matplotlib.pyplot as plt
from numpy import sum, pi, exp, sqrt, cos
import csv
import pandas as pd
from pandas.core.frame import DataFrame


def compare_Best(news,olds):
        # -1代表需要做互換
        # news=news[1]
        # olds=olds[1]
        # print(news)
        # print(olds)
        if news[0]==True and olds[0] == True and news[2]>olds[2]:
            return -1
        if news[0]==True and olds[0] == False:
            return -1
        if news[0]==False and olds[0] == False and news[3] < olds[3]:
            return -1
        return 1

def compare_Best_bool(news,olds):
        
        if news[0]==True and olds[0] == True and news[2]>olds[2]:
            # print(True)
            return True
        if news[0]==True and olds[0] == False:
            # print(True)
            return True
        if news[0]==False and olds[0] == False and news[3] < olds[3]:
            # print(True)
            return True
        # print(False)
        return False

def f1(x):

    return np.sum(x ** 2)
    # return sum(num**2 for num in x)

def f2(x):
    x=np.fabs(x)
    return np.sum(x)+np.prod(x)
    # return sum(num**2 for num in x)

def f3(x):
    # print(x)
    dim = x.shape[0]
    ans=0
    for i in range(dim):
        ans += np.sum(x[:i+1])**2
    return ans

def f2_compare(news,olds):
    if news > olds:
        return 1
    return -1


def f2_compare_Best_bool(news,olds):
    if news < olds:
        return True
    return False

def ResultToCsv(Alg):
    # np.savetxt(f'{Alg.__class__.__name__}.csv',Alg.best,delimiter=",")

    np.savetxt(f'{Alg.__class__.__name__}.csv',Alg.convergence,delimiter=",")

def csv_to_LO(data):
    
    dfs=[]
    for np_array in data:
        df = pd.DataFrame(np_array)
        dfs.append(df)
    result = pd.concat(dfs)
    result.to_csv('postion_HHO_SMA.csv',index=False)
    
    
def plotBox(Algs):
    alg_list = []
    for Alg in Algs:
        print(Alg.avg_bestIndividual) 
        reshape_Alg =np.reshape(Alg.avg_bestIndividual,(2,base_numbers,ue_numbers))
        check_conectional = (reshape_Alg[0] > 500)
        rate_table=calculator_throughtput(reshape_Alg,check_conectional)
        every_ue_rate=np.sum(rate_table,axis=0)
        print(every_ue_rate)
        alg_list.append(every_ue_rate)
    fig, ax = plt.subplots()
    bp = ax.boxplot(alg_list, showfliers=False)
    positions = [i+1 for i in range(len(Algs))]
    for i in range(len(alg_list)):
        y = alg_list[i]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=8, color='black')
    # plt.boxplot(every_ue_rate)
    ax.set_xticklabels(['GA','SMA','HHO_DE','HHO','HHO_chaos','HHO_SMA'])
    plt.savefig("box.jpg")
    # plt.show()
    plt.close()
    csv_data = DataFrame(alg_list)
    csv_data =csv_data.T
    csv_data.rename(columns={0:'GA',1:'SMA',2:'HHO_DE',3:'HHO',4:'HHO_chaos',5:'HHO_SMA'},inplace=True)
    print(csv_data)
    csv_data.to_csv('throghput_results.csv')
    

if __name__ == "__main__":
    # 毫米波網路覆蓋區域大小
    area_size = 20
    
   

    function=f2
    dimension=30
    # dimension = 30
    iteration=1000
    problem_size=30
    lb=-10
    ub=10
    compare_func = f2_compare
    parents_portion = 0.3
    mutation_prob = 0.1
    crossover_prob = 0.7
    elit_ratio = 0.01
    cross_type = 'uniform'
    Start_run_times=0
    End_run_times=30
    random_seed_list = random.sample(range(1, 1000), End_run_times)
    
    # Ga1 =GA(function,dimension,iteration,problem_size,lb,ub,compare_func,f2_compare_Best_bool,parents_portion,mutation_prob,crossover_prob,elit_ratio,cross_type)
    # Ga1.mutil_run(Start_run_times,End_run_times,random_seed_list)
    # hhoSMA_DE_1 = HHO_SMA_DE(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    # hhoSMA_DE_1.mutil_run(16,End_run_times,random_seed_list)
    
    
    
    # hhoSMA_1 = HHO_SMA(function,dimension,iteration,problem_size,lb,ub,compare_func,f2_compare_Best_bool,ue_numbers,base_numbers)
    hho_1 = HHO(function,dimension,500,problem_size,lb,ub,compare_func,f2_compare_Best_bool)
    hho_1.mutil_run(Start_run_times,End_run_times,random_seed_list,'f2')
    
    # hhoSMA_chaos_1 = HHO_DE_Chaos(function,dimension,iteration,problem_size,lb,ub,compare_func,f2_compare_Best_bool)
    # hhoSMA_chaos_1.mutil_run(Start_run_times,End_run_times,random_seed_list)
    # hhoSMA_1 = HHO_SMA(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    # hhoSMA_1.mutil_run(Start_run_times,End_run_times,random_seed_list)
    OriginalSMA_1=OriginalSMA(function,dimension,1000,problem_size,lb,ub,compare_func,f2_compare_Best_bool)
    OriginalSMA_1.mutil_run(Start_run_times,End_run_times,random_seed_list,'f2')
    # fig_EE,ax_EE = plt.subplots()
    # fig_violation,ax_violation = plt.subplots()
    # # ax_EE.plot(Ga1.avg_convergence,label='GA')
    # ax_EE.plot(OriginalSMA_1.avg_convergence,label='SMA')
    # ax_EE.plot(hhoSMA_DE_1.avg_convergence,label='HHO_DE')
    # ax_EE.plot(hho_1.avg_convergence,label='HHO')
    # ax_EE.plot(hhoSMA_chaos_1.avg_convergence,label='HHO_chaos')
    # ax_EE.plot(hhoSMA_1.avg_convergence,label='HHO_SMA')
    # fig_EE.legend()
    # ax_violation.plot(Ga1.avg_constrained_violation_curve,label='GA')
    # ax_violation.plot(OriginalSMA_1.avg_constrained_violation_curve,label='SMA')
    # ax_violation.plot(hhoSMA_DE_1.avg_constrained_violation_curve,label='HHO_DE')
    # ax_violation.plot(hho_1.avg_constrained_violation_curve,label='HHO')
    # ax_violation.plot(hhoSMA_chaos_1.avg_constrained_violation_curve,label='HHO_chaos')
    # ax_violation.plot(hhoSMA_1.avg_constrained_violation_curve,label='HHO_SMA')
    
    # fig_violation.legend()
    # fig_EE.savefig("EE.jpg")
    # fig_violation.savefig("violation.jpg")
    # plotBox([Ga1,OriginalSMA_1,hhoSMA_DE_1,hho_1,hhoSMA_chaos_1,hhoSMA_1])
    # fig_violation.savefig("HHO_inv.jpg")
    # fig_EE.savefig("HHO_EE.jpg")
    # # csv_to_LO(hhoSMA_1.posRecord)
    # # csv_to_LO(hho_1.posRecord)
    
    # # ResultToCsv(hhoSMA_1)
    # # # ResultToCsv(hho_1)
    # # # ResultToCsv(OriginalSMA_1)
    # # # ResultToCsv(Ga1)
    
    # plotBox([hhoSMA_1])
    # # plt.plot(Ga1.convergence,label='GA')
    # # plt.plot(OriginalSMA_1.convergence,label='BaseSMA')
    
    
    # plt.close()
    
    
    
    
