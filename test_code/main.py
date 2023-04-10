import math
import numpy as np
import HHO
import random
import matplotlib.pyplot as plt
from numpy import sum, pi, exp, sqrt, cos
import csv
W = (10**9) ##(Hz)
alpha=2 ##(Path loss exponent)
lambd = 0.005 #(m)
Min_Rate = 8*(10**8) #(bit/s)
N_0=-174 #(dBm/Hz)
Min_Power = 0 #(mW)
Max_power = 1000 #(mW)
base_numbers = 4
ue_numbers = 20
constrained_num=6

def func_sum(solution):
    # solution = solution.reshape(2,base_numbers,ue_numbers)
    # print(np.shape(solution))
    # for i in solution:
    #     print(i)
    return sum(solution ** 2)

def find_MaxEE(X):
    three_D_X = X.reshape(2,base_numbers,ue_numbers)
    NP_three_D_X=np.array(three_D_X)
    Feasible = False
    ## three_D_X is 3D matrix ,First Dimension is distinguisih connectivity & Power
    ## Second Dimension is diffient BaseSation
    ## Thrid Dimension is diffient user
    Check_constrained = [False for i in range(constrained_num)]

    ## C1 constrained solution
    if np.all((NP_three_D_X[0] >= 0) & (NP_three_D_X[0] <= 1000)):
        Check_constrained[0] = True


    ## C2 constrained solution
    if np.all((NP_three_D_X[1] >= 0) & (NP_three_D_X[1] <= 1000)):
        Check_constrained[1] = True

    ## C3 constrained solution
    check_conectional = (NP_three_D_X[0] > 500)
    if np.all(np.sum(check_conectional,axis=1) > 0):
        Check_constrained[2] = True

    ## C4 constrained solution
    if np.all(np.sum(check_conectional,axis=0) > 0):
        Check_constrained[3] = True

    ## C5 constrained solution
    Base_ue_ThroughtpusTable = calculator_throughtput(NP_three_D_X,check_conectional)
    if  np.all(np.sum(Base_ue_ThroughtpusTable,axis=0) >= Min_Rate):
        Check_constrained[4] = True

    ## C6 constrained solution
    if  np.all(np.sum(NP_three_D_X[1]*check_conectional,axis=1)<= Max_power) :
        Check_constrained[5] = True

    Energy_efficient = np.sum(Base_ue_ThroughtpusTable) / (np.sum(NP_three_D_X[1]*check_conectional) + 10E-10)
    constrained_violation = (   
        max(np.sum((1-np.sum(check_conectional,axis=0))),0) +   
        max(np.sum((1-np.sum(check_conectional,axis=1))),0) +
        max(np.sum(Min_Rate - np.sum(Base_ue_ThroughtpusTable,axis=0))/(10**6),0)+
        np.sum(np.maximum( np.sum(NP_three_D_X[1]*check_conectional,axis=1) - Max_power,0))
        )
    test_num = (max(np.sum((1-np.sum(check_conectional,axis=0))),0),
                max(np.sum((1-np.sum(check_conectional,axis=1))),0) ,
                max(np.sum(Min_Rate - np.sum(Base_ue_ThroughtpusTable,axis=0)),0),
                np.sum(np.maximum( np.sum(NP_three_D_X[1]*check_conectional,axis=1) - Max_power,0)),
                np.sum(NP_three_D_X[1]*check_conectional,axis=1)- Max_power 
                )
    
    # print("-----------------------------------------------")
    # print(max(np.sum((1-np.sum(check_conectional,axis=0))),0) )
    # print(max(np.sum((1-np.sum(check_conectional,axis=1))),0))
    # print(max(np.sum(Min_Rate - np.sum(Base_ue_ThroughtpusTable,axis=0))/Min_Rate,0))
    # print(max(np.sum(np.sum(NP_three_D_X[1]*check_conectional,axis=1)),0))
    if np.all(Check_constrained[:5]):
        Feasible = True
    return (Feasible,Check_constrained,Energy_efficient,constrained_violation,test_num)


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
    


# class BS:

#     def __init__(self, x, y, power):
#         self.x = x
#         self.y = y
#         self.power = power
#         self.connected_UEs = []
    
#     def connect_UE(self, UE):
#         self.connected_UEs.append(UE)
    
#     def get_connected_UE_count(self):
#         return len(self.connected_UEs)
    
#     def get_power_usage(self):
#         usage = 0
#         for ue in self.connected_UEs:
#             usage += ue.power
#         return usage
    
#     def get_throughput(self, UE):
#         # 根据UE和BS之间的距离等信息计算吞吐量
#         return throughput
    

# class UE:
#     def __init__(self, x, y, power):
#         self.x = x
#         self.y = y
#         self.power = power
#         self.connected_BSs = []
    
#     def connect_BS(self, BS):
#         self.connected_BSs.append(BS)
    
#     def get_connected_BS_count(self):
#         return len(self.connected_BSs)
    
#     def get_power_usage(self):
#         return self.power


if __name__ == "__main__":
    
    

    # 毫米波網路覆蓋區域大小
    area_size = 20
    
    # 建立4個基站位置的列表
    bs_positions = [(6,6),(14,6),(6,14),(14,14)]
    # for i in range(base_numbers):
    #     x = random.uniform(0, area_size)
    #     y = random.uniform(0, area_size)
    #     bs_positions.append((x, y))

    # 建立10個用戶設備位置的列表
    ue_positions = []
    for i in range(ue_numbers):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        ue_positions.append((x, y))

    with open("deploy.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(bs_positions)
        writer.writerow(ue_positions)
    dim=2*ue_numbers*base_numbers
    SearchAgents_no=500
    lb=0
    ub=1000
    Max_iter=100

    lb = [0]
    ub = [1000]
    problem_size = 2*ue_numbers*base_numbers

    obj_func = find_MaxEE
    verbose = True
    epoch = 100
    pop_size = 500

    for i in range(0,1):

        HHO_value=HHO.HHO(find_MaxEE, lb, ub, dim, SearchAgents_no, Max_iter)
    #     impove_HHO_value=HHO_combin.HHO(find_MaxEE, lb, ub, dim, SearchAgents_no, Max_iter)
        
        
    #     # md3 = SMA_combine.OriginalSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
    #     # best_pos2, best_fit2, list_loss2,compare2 = md3.train()
    #     md2 = OriginalSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
    #     best_pos2, best_fit2, list_loss2,compare2 = md2.train()

    #     md1 = BaseSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
    #     best_pos1, best_fit1, list_loss1, compare = md1.train()
    #     np_list_loss1=np.array(list_loss1)
    #     np_list_loss2=np.array(list_loss2)
    #     if i==0:
    #         ALL_HHO = HHO_value.convergence
    #         ALL_impove_HHO = impove_HHO_value.convergence
    #         ALL_np_list_loss1 = np_list_loss1
    #         ALL_np_list_loss2 = np_list_loss2
    #     else:
    #         ALL_HHO = (ALL_HHO+HHO_value.convergence)/2
    #         ALL_impove_HHO = (ALL_impove_HHO+impove_HHO_value.convergence)/2
    #         ALL_np_list_loss1 = (ALL_np_list_loss1+np_list_loss1)/2
    #         ALL_np_list_loss2 = (ALL_np_list_loss2+np_list_loss2)/2


    # plt.plot(ALL_HHO,label='HHO')
    # plt.plot(ALL_impove_HHO,label='impove_HHO')
    # plt.plot(ALL_np_list_loss1 ,label='Original SMA')
    # plt.plot(ALL_np_list_loss2,label='Base SMA')
    # plt.legend()
    # plt.savefig("SMA.jpg")
    # plt.show()


