import numpy as np
import math
import random
import sys
import matplotlib.pyplot as plt
N=-174
X=10**((N-30)/10)
SINR=1/(X*(10**9))
# print((10**9) * np.log2(1 + SINR)/8/(10**9))
W = 10**9 ##(Hz)
alpha=2 ##(Path loss exponent)
lambd = 0.005 #(m)
Min_Rate = 10**9 #(bit/s)
N_0=-174 #(dBm/Hz)
Min_Power = 0 #(mW)
Max_power = 10000 #(mW)
base_numbers = 4
ue_numbers = 20
constrained_num=6

import csv

def throughtput(num_power,num_distance):    
    receive_power=P_ij_r(num_power*0.001,lambd,num_distance,alpha,Rayleigh_fading())
    throughtput = R_ij(W,1, SNR_ij(receive_power,W,N_0))
    return throughtput


def calculator_throughtput(NP_three_D_X,check_conectional,bs_positions,ue_positions):
    throughtput_table = np.zeros(np.shape(check_conectional))
    for ue in range(ue_numbers):
        for base in range(base_numbers):
            if check_conectional[base][ue]:
                receive_power=P_ij_r(NP_three_D_X[1][base][ue]*0.001,lambd,euclidean_distance(bs_positions[base], ue_positions[ue]),alpha,Rayleigh_fading())
                throughtput = R_ij(W,np.sum(check_conectional,axis=1)[base], SNR_ij(receive_power,W,N_0))
                throughtput_table[base][ue] = throughtput
                
    return  throughtput_table
        

def Rayleigh_fading():
    x = np.random.normal(loc=0,scale=1)
    y = np.random.normal(loc=0,scale=1)
    z =  np.sqrt(0.5) * (x + 1j*y)
    # return np.abs(z)     
    return 1           


def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def R_ij(W, x_ij, SNR_ij):
    return (W/x_ij)* np.log2(1 + SNR_ij)

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
    
def read_csv():
    try:
        with open('deplo.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            float_tuples = []
            for row in data:
                float_tuple_row = []
                for item in row:
                    # 將逗號分隔的字串分割為兩個字串
                    items = item.split(',')
                    # 將兩個字串轉換為float數字
                    float_num1 = float(items[0].strip('()'))
                    float_num2 = float(items[1].strip('()'))
                    # 將兩個float數字組成tuple
                    float_tuple_row.append((float_num1, float_num2))
                float_tuples.append(tuple(float_tuple_row))
    except FileNotFoundError:
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
        float_tuples = [bs_positions,ue_positions]
    return float_tuples
    
if '__main__' == __name__:
    # deploy=read_csv()
    # bs_positions = deploy[0]
    # ue_positions = deploy[1]
    # connect_power=[1000 for i in range(2*base_numbers*ue_numbers)]
    # NP_three_D_X=np.array(connect_power)
    # three_D_X = NP_three_D_X.reshape(2,base_numbers,ue_numbers)
    # NP_three_D_X=np.array(three_D_X)
    # check_conectional = (NP_three_D_X[0] > 500)
    
    # ans=calculator_throughtput(NP_three_D_X,check_conectional,bs_positions,ue_positions)
    
    # ans=np.array(ans)
    
    np.set_printoptions(threshold=sys.maxsize)
    # print(ans)
    # print(f'True : {np.sum(tureFlase_ans!=0)}')
    # print(f'False : {np.sum(tureFlase_ans==0)}')
    conecation = np.linspace(1,10,10)
    num_power = 1000
    num_distance = np.linspace(1,20,20)
    conecation,num_distance = np.meshgrid(conecation,num_distance)
    ans=throughtput(num_power/conecation,num_distance)
    ans=ans*conecation
    ans[ans<(10**9)] = np.nan
    print(np.max(ans))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.imshow(tureFlase_ans, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    ax.contourf(conecation,num_distance,ans)
    plt.show()
    # print(np.shape(ans))
    # print(np.sum(ans,axis=0))
    # print(np.sum(NP_three_D_X[1],axis=1))
    # print(np.sum(ans)/np.sum(check_conectional*NP_three_D_X[1]))
