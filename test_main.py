
import numpy as np
from Alg import HHO,BaseSMA,OriginalSMA,HHO_SMA,GA,HHO_SMA_ch,HHO_SMA_Chaos,HHO_SMA_DE
import math
import Alg
import random
import matplotlib.pyplot as plt
from numpy import sum, pi, exp, sqrt, cos
import csv
import pandas as pd
from pandas.core.frame import DataFrame
W = (10**9) ##(Hz)
alpha=2 ##(Path loss exponent)
lambd = 0.005 #(m)
Min_Rate = (10**8) #(bit/s)
N_0=-174 #(dBm/Hz)
Min_Power = 0 #(mW)
Max_power = 1000 #(mW)
base_numbers = 4
ue_numbers = 5
constrained_num=6

def find_MaxEE(X):
    # three_D_X = X.reshape(2,base_numbers,ue_numbers)
    NP_three_D_X=np.reshape(X,(2,base_numbers,ue_numbers))
    # NP_three_D_X=np.array(three_D_X)
    Feasible = False
    ## three_D_X is 3D matrix ,First Dimension is distinguisih connectivity & Power
    ## Second Dimension is diffient BaseSation
    ## Thrid Dimension is diffient user
    Check_constrained = [False for i in range(constrained_num)]

    ## C1 constrained solution
    if np.all((NP_three_D_X[0] >= 0) & (NP_three_D_X[0] <= 1000)):
        Check_constrained[0] = True


    ## C2 constrained solution
    # 0
    if np.all((NP_three_D_X[1] >= 0) & (NP_three_D_X[1] <= 1000)):
        Check_constrained[1] = True

    ## C3 constrained solution
    check_conectional = (NP_three_D_X[0] > 500)
    if np.all(np.sum(check_conectional,axis=1) > 0):
        Check_constrained[2] = True
    Zero_mask_connect_BS = np.where(np.sum(check_conectional,axis=1) <= 0)

    ## C4 constrained solution
    if np.all(np.sum(check_conectional,axis=0) > 0):
        Check_constrained[3] = True
    Zero_mask_connect_UE=np.where((np.sum(check_conectional,axis=1) <= 0))

    ## C5 constrained solution
    Base_ue_ThroughtpusTable = calculator_throughtput(NP_three_D_X,check_conectional)
    if  np.all(np.sum(Base_ue_ThroughtpusTable,axis=0) >= Min_Rate):
        Check_constrained[4] = True
    rate_inadeuate_mask=np.where((np.sum(Base_ue_ThroughtpusTable,axis=0) < Min_Rate))

    ## C6 constrained solution
    if  np.all(np.sum(NP_three_D_X[1]*check_conectional,axis=1)<= Max_power) :
        Check_constrained[5] = True
    Over_Power_mask=np.where((np.sum(NP_three_D_X[1]*check_conectional,axis=1)> Max_power))

    Energy_efficient = np.sum(Base_ue_ThroughtpusTable) / (np.sum(NP_three_D_X[1]*check_conectional) + 10E-10)
    constrained_violation = (
        (not Check_constrained[0])*10000+
        (not Check_constrained[1])*10000+      
        np.sum(np.maximum( (1-np.sum(check_conectional,axis=0)) ,0) )*10000 +   
        np.sum(np.maximum( (1-np.sum(check_conectional,axis=1)) ,0) )*10000 +
        np.sum(np.maximum( ( Min_Rate - np.sum(Base_ue_ThroughtpusTable,axis=0)) / (10**6) , 0))+
        np.sum(np.maximum( np.sum(NP_three_D_X[1]*check_conectional,axis=1) - Max_power, 0 )*100)
        )
    test_num = ((not Check_constrained[0])*10000,
                (not Check_constrained[1])*10000,
                np.sum(np.maximum( (1-np.sum(check_conectional,axis=0)) ,0) ),   
                np.sum(np.maximum( (1-np.sum(check_conectional,axis=1)) ,0) ),
                np.sum(np.maximum( ( Min_Rate - np.sum(Base_ue_ThroughtpusTable,axis=0)) / (10**6) , 0)),
                np.sum(np.maximum( np.sum(NP_three_D_X[1]*check_conectional,axis=1) - Max_power, 0 )),
                np.sum(NP_three_D_X[1]*check_conectional,axis=1)- Max_power 
                )
    mask = np.array([Zero_mask_connect_BS,Zero_mask_connect_UE,rate_inadeuate_mask,Over_Power_mask])
    # every_ue_throughput=0
    # for i in range(Base_ue_ThroughtpusTable.size):
    #     if Base_ue_ThroughtpusTable[i] != 0:
    #         if Min_Rate / Base_ue_ThroughtpusTable[i] > 1:
    #             every_ue_throughput+= Min_Rate / Base_ue_ThroughtpusTable[i]
    #         else:
    #             pass
    #     else:
    #         every_ue_throughput += Min_Rate

    
    # constrained_violation = (   
    #     np.sum(np.maximum( (1-np.sum(check_conectional,axis=0)) , 0 ) )+
    #     np.sum(np.maximum( (1-np.sum(check_conectional,axis=1)) , 0 ) )+
    #     every_ue_throughput +
    #     np.maximum((np.sum(NP_three_D_X[1]*check_conectional,axis=1) / Max_power), 1)

    #     )
    # test_num = (
    #     np.sum(np.maximum( (1-np.sum(check_conectional,axis=0)) , 0 ) ),
    #     np.sum(np.maximum( (1-np.sum(check_conectional,axis=1)) , 0 ) ),
    #     every_ue_throughput ,
    #     np.maximum((np.sum(NP_three_D_X[1]*check_conectional,axis=1) / Max_power), 1)

    #     )
    
    if np.all(Check_constrained):
        Feasible = True
    return (Feasible,Check_constrained,Energy_efficient,constrained_violation,test_num,mask)

def calculator_throughtput(NP_three_D_X,check_conectional):
    throughtput_table = np.zeros(np.shape(check_conectional))
    for ue in range(ue_numbers):
        for base in range(base_numbers):
            if NP_three_D_X[0][base][ue] > 500:
                receive_power=P_ij_r(NP_three_D_X[1][base][ue]*0.001,lambd,euclidean_distance(bs_positions[base], ue_positions[ue]),alpha,Rayleigh_fading())
                throughtput = R_ij(W,np.sum(check_conectional,axis=1)[base], SNR_ij(receive_power,W,N_0))
                throughtput_table[base][ue] = throughtput
    return  throughtput_table
        

def Rayleigh_fading():
    x = np.random.normal(loc=0,scale=1)
    y = np.random.normal(loc=0,scale=1)
    z =  np.sqrt(0.5) * (x + 1j*y)
    return np.abs(z)                


def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def R_ij(W, x_ij, SNR_ij):
    return (W/x_ij) * np.log2(1 + SNR_ij)

def SNR_ij(P_ij_r, W, N_0):
    N_0_to_W = 10**((N_0 - 30) / 10)
    return P_ij_r / (W * N_0_to_W)

def P_ij_r(P_ij_t, lambd, d_ij, alpha, h_ij):
    return P_ij_t * 1 * (lambd / (4 * np.pi))**2 * (d_ij**(-alpha)) * h_ij*(10**(-3))

def G(psi, Theta):
    if abs(psi) <= Theta:
        return 1
    else:
        return 0

def compare_Best(news,olds):
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

def f2(x):
    ans=1
    for i in x:
        ans*=abs(i)
    return sum(abs(num) for num in x) + ans
    # return sum(num**2 for num in x)

def f2_compare(news,olds):
    if news > olds:
        return 1
    return -1


def f2_compare_Best_bool(news,olds):
    if news > olds:
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
    np.random.seed(000)
    random.seed(000)
    # 毫米波網路覆蓋區域大小
    area_size = 20
    
    # 建立4個基站位置的列表
    bs_positions = [(6,6),(14,6),(6,14),(14,14)]
    # bs_positions = [(5,5),(5,10),(5,15),(10,5),(10,10),(10,15),(15,5),(15,10),(15,15)]
    ue_positions = []
    for i in range(ue_numbers):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        ue_positions.append((x, y))

    function=find_MaxEE
    dimension=2*ue_numbers*base_numbers
    # dimension = 30
    iteration=10
    problem_size=100
    lb=0
    ub=1000
    compare_func = compare_Best
    parents_portion = 0.3
    mutation_prob = 0.1
    crossover_prob = 0.7
    elit_ratio = 0.01
    cross_type = 'uniform'
    run_times=25
    random_seed_list = random.sample(range(1, 1000), run_times)
    
    Ga1 =GA(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool,parents_portion,mutation_prob,crossover_prob,elit_ratio,cross_type)
    Ga1.mutil_run(run_times,random_seed_list)
    OriginalSMA_1=OriginalSMA(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    OriginalSMA_1.mutil_run(run_times,random_seed_list)
    
    hhoSMA_DE_1 = HHO_SMA_DE(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    hhoSMA_DE_1.mutil_run(run_times,random_seed_list)
    # hhoSMA_1 = HHO_SMA(function,dimension,iteration,problem_size,lb,ub,compare_func,f2_compare_Best_bool,ue_numbers,base_numbers)
    hho_1 = HHO(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    hho_1.mutil_run(run_times,random_seed_list)
    
    hhoSMA_chaos_1 = HHO_SMA_Chaos(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    hhoSMA_chaos_1.mutil_run(run_times,random_seed_list)
    hhoSMA_1 = HHO_SMA(function,dimension,iteration,problem_size,lb,ub,compare_func,compare_Best_bool)
    hhoSMA_1.mutil_run(run_times,random_seed_list)

    fig_EE,ax_EE = plt.subplots()
    fig_violation,ax_violation = plt.subplots()
    ax_EE.plot(Ga1.avg_convergence,label='GA')
    ax_EE.plot(OriginalSMA_1.avg_convergence,label='SMA')
    ax_EE.plot(hhoSMA_DE_1.avg_convergence,label='HHO_DE')
    ax_EE.plot(hho_1.avg_convergence,label='HHO')
    ax_EE.plot(hhoSMA_chaos_1.avg_convergence,label='HHO_chaos')
    ax_EE.plot(hhoSMA_1.avg_convergence,label='HHO_SMA')
    fig_EE.legend()
    ax_violation.plot(Ga1.avg_constrained_violation_curve,label='GA')
    ax_violation.plot(OriginalSMA_1.avg_constrained_violation_curve,label='SMA')
    ax_violation.plot(hhoSMA_DE_1.avg_constrained_violation_curve,label='HHO_DE')
    ax_violation.plot(hho_1.avg_constrained_violation_curve,label='HHO')
    ax_violation.plot(hhoSMA_chaos_1.avg_constrained_violation_curve,label='HHO_chaos')
    ax_violation.plot(hhoSMA_1.avg_constrained_violation_curve,label='HHO_SMA')
    
    fig_violation.legend()
    fig_EE.savefig("EE.jpg")
    fig_violation.savefig("violation.jpg")
    plotBox([Ga1,OriginalSMA_1,hhoSMA_DE_1,hho_1,hhoSMA_chaos_1,hhoSMA_1])
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
    
    
    
    
