import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import random
import math
import csv
import pickle

# W = (10**9) ##(Hz)
# alpha=2 ##(Path loss exponent)
# lambd = 0.005 #(m)
# Min_Rate = (10**8) #(bit/s)
# N_0=-174 #(dBm/Hz)
# Min_Power = 0 #(mW)
# Max_power = 1000 #(mW)
# base_numbers = 5
# ue_numbers = 20
# constrained_num=6

def load_csv(Folder_Path):
    convergence=np.loadtxt(f'{Folder_Path}/convergence.csv')
    constrained_violation_curve=np.loadtxt(f'{Folder_Path}/constrained_violation_curve.csv')
    bestIndividual=np.loadtxt(f'{Folder_Path}/bestIndividual.csv')
    return convergence,constrained_violation_curve,bestIndividual

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

def save_fitness_params(BS_FWA_distance,carrier_frequency,Antenna_gain,LOS_shadow,NLOS_shadow,N0,transmit_bandwidth):
        BS_FWA_distance = BS_FWA_distance
        BS_FWA_LOS=32.4+21*np.log10(BS_FWA_distance)+20*np.log10(carrier_frequency) #
        BS_FWA_NLOS=32.4+31.9*np.log10(BS_FWA_distance)+20*np.log10(carrier_frequency)

        BS_FWA_Gain_LOS_shdow = Antenna_gain - BS_FWA_LOS - LOS_shadow
        BS_FWA_Gain_NLOS_shdow = Antenna_gain - BS_FWA_NLOS - NLOS_shadow

        BS_FWA_Gain_LOS_shdow_multi =10**((BS_FWA_Gain_LOS_shdow)/10) #9 becuz noise power is to small that can't use in float

        BS_FWA_Gain_NLOS_shdow_multi =10**((BS_FWA_Gain_NLOS_shdow)/10) #9 becuz noise power is to small that can't use in float

        N0_mW = 10**((N0)/10)

        BS_FWA_Gain_NLOS_shdow_N0_multi = BS_FWA_Gain_NLOS_shdow_multi / N0_mW
        BS_FWA_Gain_LOS_shdow_N0_multi = BS_FWA_Gain_LOS_shdow_multi / N0_mW
        
        BS_FWA_Gain_NLOS_shdow_N0_multi = BS_FWA_Gain_NLOS_shdow_N0_multi
        BS_FWA_Gain_LOS_shdow_N0_multi = BS_FWA_Gain_LOS_shdow_N0_multi

        return BS_FWA_Gain_LOS_shdow_N0_multi

def plotBox(Alg_bestIndividual):
    Antenna_gain=18 #dBi 
    LOS_shadow = 4 #dBi
    NLOS_shadow = 18 #dBi
    transmit_bandwidth = 4e+8
    carrier_frequency = 28 # GHz
    N0=-83.02 #dBm
    save_fitness_params(BS_FWA_distance,carrier_frequency,Antenna_gain,LOS_shadow,NLOS_shadow,N0,transmit_bandwidth)
    reshape_Alg =np.reshape(Alg_bestIndividual,(2,base_numbers,ue_numbers))
    max_indices = np.argmax(reshape_Alg[0], axis=0)
    check_conectional = np.zeros((base_numbers, ue_numbers), dtype=bool)
    check_conectional[max_indices, np.arange(ue_numbers)] = True

    transmit_bandwidth = 4e+8
    rate_table=transmit_bandwidth * np.log2(1+(check_conectional*BS_FWA_distance*(reshape_Alg[1])))
    # rate_table=calculator_throughtput(reshape_Alg,check_conectional)
    every_ue_rate=np.sum(rate_table,axis=0)
    return every_ue_rate
    # alg_list.append(every_ue_rate)
    
    # bp = ax.boxplot(alg_list, showfliers=False)
    # positions = [i+1 for i in range(len(Algs))]
    # for i in range(len(alg_list)):
    #     y = alg_list[i]
    #     x = np.random.normal(i + 1, 0.04, size=len(y))
    #     ax.scatter(x, y, alpha=0.5, s=8, color='black')
    # # plt.boxplot(every_ue_rate)
    # ax.set_xticklabels(['GA','SMA','HHO_DE','HHO','HHO_chaos','HHO_SMA'])
    # plt.savefig("box.jpg")
    # # plt.show()
    # plt.close()
    # csv_data = DataFrame(alg_list)
    # csv_data =csv_data.T
    # csv_data.rename(columns={0:'GA',1:'SMA',2:'HHO_DE',3:'HHO',4:'HHO_chaos',5:'HHO_SMA'},inplace=True)
    # print(csv_data)
    # csv_data.to_csv('throghput_results.csv')


if __name__ == "__main__":
    
    np.random.seed(000)
    random.seed(000)

    # 毫米波網路覆蓋區域大小
    
    area_size = 20
    
    # # 建立4個基站位置的列表
    # bs_positions = [(6,6),(14,6),(6,14),(14,14)]
    # # bs_positions = [(5,5),(5,10),(5,15),(10,5),(10,10),(10,15),(15,5),(15,10),(15,15)]
    # ue_positions = []
    # for i in range(ue_numbers):
    #     x = random.uniform(0, area_size)
    #     y = random.uniform(0, area_size)
    #     ue_positions.append((x, y))
    with open('5BS_20FWA_coords.pkl','rb') as file:
        loadad_data = pickle.load(file)
    
    BS_FWA_distance=loadad_data['FWA_BS_distance']
    base_numbers=BS_FWA_distance.shape[0]
    ue_numbers=BS_FWA_distance.shape[1]

    # Alg_list =  ['GA','HHO','HHO_SMA','HHO_SMA_Chaos','HHO_SMA_DE','OriginalSMA']
    # Alg_list =  ['GA','HHO','HHO_SMA','HHO_SMA_Chaos','HHO_SMA_DE','OriginalSMA','HHO_DE_Chaos','Random']
    Alg_list =  ['HHO','GA','HHO_SMA','HHO_DE_Chaos/0.3']
    ##,'OriginalSMA'
    # CR=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # CR=[0.3]
    # for index,i in enumerate(CR):
    #     CR[index]=f'{Alg_list[0]}/{i}'
    
    Init_Path = './2023-09-18'
    StartTime = 0
    EndTime = 50
    fig_EE,ax_EE = plt.subplots()
    fig_violation,ax_violation = plt.subplots()
    fig_box, ax_box = plt.subplots()
    fig_bar, ax_bar = plt.subplots()
    fig_barstd, ax_barstd = plt.subplots()
    box_list=[]
    boxname_list=[]
    std_EE = []
    avg_EE=[]
    std_violation = []
    for index,Alg in enumerate(Alg_list):
        EE_Alg = []
        violation_Alg = []
        for Time in range(StartTime,EndTime):
            if Time == 0:
                avg_convergence,avg_constrained_violation_curve,bestIndividual=\
                load_csv(f'{Init_Path}/{Alg}/{Time}')
                avg_throughput=plotBox(bestIndividual)
                EE_Alg.append(avg_convergence[-1])
                violation_Alg.append(avg_constrained_violation_curve[-1])
            else:
                convergence,constrained_violation_curve,bestIndividual=\
                load_csv(f'{Init_Path}/{Alg}/{Time}')
                EE_Alg.append(convergence[-1])
                violation_Alg.append(constrained_violation_curve[-1])
                avg_convergence = (avg_convergence + convergence)/2
                avg_constrained_violation_curve = (avg_constrained_violation_curve + constrained_violation_curve) /2
                # avg_bestIndividual = (avg_bestIndividual + bestIndividual) /2
                avg_throughput=(plotBox(bestIndividual)+avg_throughput)/2
        avg_EE.append(np.mean(np.array(EE_Alg)))
        
        std_EE.append(np.std(np.array(EE_Alg)))
        
        std_violation.append(np.std(np.array(violation_Alg)))    
        # throughput=plotBox(avg_bestIndividual)
        
        y = avg_throughput
        x = np.random.normal(index + 1, 0.04, size=len(y))
        ax_box.scatter(x, y, alpha=0.5, s=8, color='black')
        box_list.append(y)
        boxname_list.append(f'{Alg}')
        ax_EE.plot(avg_convergence,label=f'{Alg}')
        ax_violation.plot(avg_constrained_violation_curve,label=f'{Alg}')
    ax_bar.bar([i*10 for i in range(len(avg_EE))],avg_EE,width=2.4)
    # ax_bar.legend()
    ax_bar.set_xlabel('crossover probability (%)')
    ax_bar.set_ylabel('average Energy efficiency (bit/s)/mW')
    ax_bar.set_title('compare different crossover probability')
    fig_bar.savefig(f'{Init_Path}/EE_bar.jpg')

    ax_barstd.bar([i*10 for i in range(len(std_EE))],std_EE,width=2.4)
    # ax_barstd.legend()
    ax_barstd.set_xlabel('crossover probability (%)')
    ax_barstd.set_ylabel('std Energy efficiency (bit/s)/mW')
    ax_barstd.set_title('compare different crossover probability')
    fig_barstd.savefig(f'{Init_Path}/EE_barstd.jpg')
    fig_box.boxplot(box_list, showfliers=False,labels=boxname_list)
    fig_box.legend()
    fig_EE.legend()
    fig_violation.legend()
    fig_box.savefig(f'{Init_Path}/box.jpg')
    fig_EE.savefig(f'{Init_Path}/EE.jpg')
    fig_violation.savefig(f'{Init_Path}/violation.jpg')
    # save throughput csv
    csv_data = DataFrame(box_list)
    csv_data =csv_data.T
    csv_data.rename(columns={index:name for index,name in enumerate(boxname_list)},inplace=True)
    csv_data.to_csv(f'{Init_Path}/throghput_results.csv')
    csv_data = DataFrame(std_EE)
    csv_data =csv_data.T
    csv_data.rename(columns={index:name for index,name in enumerate(boxname_list)},inplace=True)
    csv_data.to_csv(f'{Init_Path}/EE_std.csv')
    csv_data = DataFrame(std_violation)
    csv_data =csv_data.T
    csv_data.rename(columns={index:name for index,name in enumerate(boxname_list)},inplace=True)
    csv_data.to_csv(f'{Init_Path}/std_violation.csv')