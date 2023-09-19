import numpy as np
import random
W = (10**9) ##(Hz)
alpha=2 ##(Path loss exponent)
lambd = 0.005 #(m)
Min_Rate = (10**8) #(bit/s)
import math
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
    Start_run_times=0
    End_run_times=25
    random_seed_list = random.sample(range(1, 1000), End_run_times)

