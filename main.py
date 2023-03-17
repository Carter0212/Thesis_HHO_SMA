import math
import numpy as np
import HHO
import random
import matplotlib.pyplot as plt
from SMA import BaseSMA, OriginalSMA
from numpy import sum, pi, exp, sqrt, cos

W = 10**9 ##(Hz)
alpha=2 ##(Path loss exponent)
lambd = 0.005 #(m)
Min_Rate = 10**9 #(bit/s)
N_0=-174 #(dBm/Hz)
Min_Power = 0 #(mW)
Max_power = 1000 #(mW)
base_numbers = 4
ue_numbers = 10

def func_sum(solution):
    # solution = solution.reshape(2,base_numbers,ue_numbers)
    # print(np.shape(solution))
    # for i in solution:
    #     print(i)
    return sum(solution ** 2)

def find_MaxEE(X):
    three_D_X = X.reshape(2,base_numbers,ue_numbers)
    penalty = 1
    ## check have over powers of maximum
    all_power = []
    all_throughput = []
    all_EE = []
    for base in range(base_numbers):
        check_power=0
        number_ue=0
        for ue in range(ue_numbers):
            if three_D_X[0,base,ue] > 500:
                check_power+=three_D_X[1][base][ue]
                number_ue +=1
        if check_power > 1000:
            # return float("-inf")
            return 0
        all_power.append((check_power,number_ue))
    for ue in range(ue_numbers):
        check_throughput = 0
        for base in range(base_numbers):
            if three_D_X[0][base][ue] > 500:
                receive_power=P_ij_r(three_D_X[1][base][ue],lambd,euclidean_distance(bs_positions[base], ue_positions[ue]),alpha,Rayleigh_fading())
                throughtput = R_ij(W,all_power[base][1], SNR_ij(receive_power,W,N_0))
                check_throughput += throughtput
        # if check_throughput < (10**9):
        #     penalty *= (check_throughput/10**9)+0.01
            # return float("-inf")
        # elif check_throughput == 0:
        #     return 0
        all_throughput.append(check_throughput)
    sum_power =0
    for i in all_power:
        sum_power += i[0]
    sum_throughput = 0
    for i in all_throughput:
        sum_throughput += i
    if sum_power ==0:
        return 0
    # print("----------------------------------")
    # print(all_throughput)
    # print("==================================")
    # print(all_power)
    return (sum_throughput/sum_power)*penalty



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
    # number = x_ij
    # return (W / np.sum(x_ij)) * np.log2(1 + SNR_ij)
    # print(W,x_ij,SNR_ij)
    # return (W / x_ij) * np.log2(1 + SNR_ij)
    return (W) * np.log2(1 + SNR_ij)

def SNR_ij(P_ij_r, W, N_0):
    N_0_to_W = 10**((N_0 - 30) / 10)
    return P_ij_r / (W * N_0_to_W)

def P_ij_r(P_ij_t, lambd, d_ij, alpha, h_ij):
    return P_ij_t * 1 * (lambd / (4 * np.pi))**2 * d_ij**(-alpha) * h_ij*(10**(-3))

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
    bs_positions = []
    for i in range(base_numbers):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        bs_positions.append((x, y))

    # 建立10個用戶設備位置的列表
    ue_positions = []
    for i in range(ue_numbers):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        ue_positions.append((x, y))
    dim=2*ue_numbers*base_numbers
    SearchAgents_no=100
    lb=0
    ub=1000
    Max_iter=500
    # HHO.HHO(find_MaxEE, lb, ub, dim, SearchAgents_no, Max_iter)

    lb = [0]
    ub = [1000]
    problem_size = 2*ue_numbers*base_numbers
    ## if you choose this way, the problem_size can be anything you want


    ## Setting parameters
    obj_func = find_MaxEE
    verbose = True
    epoch = 500
    pop_size = 100

    md2 = OriginalSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
    best_pos2, best_fit2, list_loss2 = md2.train()
    # return : the global best solution, the fitness of global best solution and the loss of training process in each epoch/iteration
    print(np.shape(best_pos2))
    print(1/best_fit2)
    print(np.shape(list_loss2))
    list_loss2=1/np.array(list_loss2)
    plt.plot(list_loss2)
    plt.savefig("SMA.jpg")
    # plt.show()


