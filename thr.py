import numpy as np
import math
import matplotlib.pyplot as plt
# import opensimplex 

Antenna_gain=18 #dBi
# Shadow_fading=4 #dB

carrier_frequency = 28 # GHz

def DoDistance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def DoThroughput(transmit_bandwidth,SNR):
    """
    calculate throughput by shannon capacity
    Arg:
        transmit_bandwidth: unit(Hz)
        SNR : Signal-to-noise ratio(SNR) no unit
    
    return:
        Throughput
    """
    return transmit_bandwidth*np.log2(1+SNR)

def DoSNR(Conncection_Status,Recived_power,Noise_power):
    """
    calculate SNR 
    Arg:
        Recived_power : unit(w)
        Noise_power: unit(w)
        Conncection_Status :  no unit
    
    return:
        SNR
    """
    

    return (Conncection_Status*Recived_power)/Noise_power

def DoChannelModel(Transmit_power,Antenna_gain,Path_loss,Los_Probability):
    """
    calculate Recived_power by 3GPP spec
    Arg:
        Recived_power: unit (dBm)
        Transmit_power: unint (dB)
        Antenna_gain: unit (dB)
        Path_loss: unit (dB)
        Shadow_fading: unit (dB)
    
    return:
        Recived_power
    """
    if Transmit_power == float('-Inf'):
        return float('-Inf')

    if Los_Probability == 1:
        Recived_power = Transmit_power + Antenna_gain - Path_loss - 4
    else:
        Recived_power = Transmit_power + Antenna_gain - Path_loss - 8.2
    
    return Recived_power

def IndoorDoChannelModel(Transmit_power,Antenna_gain,Path_loss,Los_Probability):
    """
    calculate Recived_power by 3GPP spec
    Arg:
        Recived_power: unit (dBm)
        Transmit_power: unint (dB)
        Antenna_gain: unit (dB)
        Path_loss: unit (dB)
        Shadow_fading: unit (dB)
    
    return:
        Recived_power
    """
    if Transmit_power == float('-Inf'):
        return float('-Inf')

    if Los_Probability == 1:
        Recived_power = Transmit_power + Antenna_gain - Path_loss - 4
    else:
        Recived_power = Transmit_power + Antenna_gain - Path_loss - 8.2
    
    return Recived_power

def DoPathLoss(Los_Probability,carrier_frequency,Distance):
    """
    calculate path loss by 3GPP spec
    Arg:
        Los_Probability: unit (dB)
        carrier_frequency: unint (GHz)
        Distance: unit (m)
    
    return:
        Path_loss: unit (dB)
    """
    Decied_Conncection_Status=np.random.rand()
    if Los_Probability == 1:
        
        return 32.4+21*math.log10(Distance)+20*math.log10(carrier_frequency) ##unit(dB)
    else:
        
        return 32.4+31.9*math.log10(Distance)+20*math.log10(carrier_frequency) ##unit(dB)


def IndoorDoPathLoss(Los_Probability,carrier_frequency,Distance):
    """
    calculate path loss by 3GPP spec
    Arg:
        Los_Probability: unit (dB)
        carrier_frequency: unint (GHz)
        Distance: unit (m)
    
    return:
        Path_loss: unit (dB)
    """
    Decied_Conncection_Status=np.random.rand()
    if Los_Probability == 1:
        
        return 31.84+21.50*np.log10(Distance)+19.00*np.log10(carrier_frequency) ##unit(dB)
    else:
        
        return 32.4+31.9*np.log10(Distance)+20*np.log10(carrier_frequency) ##unit(dB)

def LosProbability(distance):
    """
    calculate Line-of-sight Probability by 3GPP spec
    Arg:
        Distance: unit (m)
    
    return:
        Line-of-sight Probability
    """
    minimum_distance =18
    if distance <= minimum_distance:
        return 1
    else:
        return (18/distance)+(np.exp(-1*distance/36)*(1-(18/distance)))

# def DoConectionStatus(Distance,TransmitPower):
#     Path_loss = DoPathLoss(1,carrier_frequency,Distance)
#     DoChannelModel(Transmit_power,Antenna_gain,Path_loss,Shadow_fading)

def dBmToWatt(dBm):
    
    if dBm == float('-Inf'):
        return 0
    return 10**((dBm-30)/10)

def wTodBm(mW):
    if mW == 0 :
        return float('-Inf')
    return 10 * np.log10(mW)+30


# def simplexnoise(seed,x_asix,y_asix):
#     simplex = opensimplex()
#     A = np.zeros([x_asix, y_asix])
#     for y in range(0, x_asix):
#         for x in range(0, y_asix):
#             value = simplex.noise2d(x,y)
#             color = int((value + 1) * 128)
#             A[x, y] = color

#     plt.imshow(A)
#     plt.show()



    


if __name__ == '__main__':

    X_rows = 100
    Y_cols = 100
    # simplexnoise(123,X_rows,Y_cols)
    
    LOS_NLOS = 1
    N0=-83.02  #dBm
    transmit_bandwidth = 1e+8
    throughput_2D_total = np.zeros((X_rows,Y_cols), dtype=float)
    distance_2D= np.empty((X_rows,Y_cols), dtype=float)
    indterval = 1
    base_satation = np.array([[15.1,20.1],[25.1,80.1],[80.5,43.2]])
    transmit_power = 0.1 #W
    powerdB = wTodBm(transmit_power)
    for base in base_satation:
        # print(base)
        # print('===========================')
        # print(throughput_2D_total)
        # print('===========================')
        throughput_2D = np.empty((X_rows,Y_cols), dtype=float)
        distance_2D= np.empty((X_rows,Y_cols), dtype=float)
        for x in range(X_rows):
            for y in range(Y_cols):
                distance = DoDistance(x*indterval,y*indterval,base[0],base[1])
                # print(f'ue_x:{x*indterval},ue_y:{y*indterval},bs_x:{base[0]},bs_y:{base[1]} , distance:{distance}')
                
                ### Indoor LOS calculation
                Indoor_LOS_Path_loss=IndoorDoPathLoss(LOS_NLOS,carrier_frequency,distance)
                Indoor_LOS_Recived_power = IndoorDoChannelModel(powerdB,Antenna_gain,Indoor_LOS_Path_loss,LOS_NLOS)
                Indoor_LOS_SNR=DoSNR(1,dBmToWatt(Indoor_LOS_Recived_power),dBmToWatt(N0))
                throughput_2D[x,y] = DoThroughput(transmit_bandwidth,Indoor_LOS_SNR)
                ### Indoor LOS calculation
        
        throughput_2D_total += throughput_2D
    levels = np.linspace(0,3000,30)
    x_axis,y_axis = np.meshgrid(range(X_rows),range(Y_cols))
    fig,ax = plt.subplots()
    throughput_2D_total=throughput_2D_total/(1e+6)
    
    thr=ax.contour(x_axis,y_axis,throughput_2D_total,levels=levels)
    
    ax.clabel(thr,inline=True,fontsize=10)
    
    ax.set_xlabel('x coordinate (m)')
    ax.set_ylabel('y coordinate (m)')
    ax.set_title('Analyze user rates at different coordubates (Mbps)\nLOS Urban Micro cell (UMi)')
    ax.set_aspect('equal')
    ax.scatter(base_satation[:,1],base_satation[:,0],c='red',marker='^',label='Base station',s=100,zorder=10)
    ax.legend()
    plt.savefig('thr.jpg')
    plt.show()        
    # for en in np.nditer(distance_2D):
    #     print(en)
    print(throughput_2D)
    exit(1)
    Distance = 100
    N0=-83.02  #dBm
    recp=-115.44
    transmit_bandwidth = 1e+8
    transmit_power = 0.2 # W
    LOS_NLOS = 1
    dis = 100
    for dis in range(1,100):
        powerdB = wTodBm(transmit_power)
        print(powerdB)
        LOS_Path_loss=DoPathLoss(LOS_NLOS,carrier_frequency,dis)
        NLOS_Path_loss=DoPathLoss(0,carrier_frequency,dis)
        Indoor_LOS_Path_loss=IndoorDoPathLoss(LOS_NLOS,carrier_frequency,dis)
        print(f'==============Distance == {dis}m ==============')
        print(f'LOS_Path_loss : {LOS_Path_loss} , NLOS_Path_loss:{NLOS_Path_loss}, Indoor_LOS_Path_loss:{Indoor_LOS_Path_loss}')
        # print(Path_loss)
        Indoor_LOS_Recived_power = IndoorDoChannelModel(powerdB,Antenna_gain,Indoor_LOS_Path_loss,LOS_NLOS)
        LOS_Recived_power=DoChannelModel(powerdB,Antenna_gain,LOS_Path_loss,LOS_NLOS)
        NLOS_Recived_power=DoChannelModel(powerdB,Antenna_gain,NLOS_Path_loss,0)
        print(f'LOS_Recived_power:{LOS_Recived_power}  ,  NLOS_Recived_power:{NLOS_Recived_power},Indoor_LOS_Recived_power : {Indoor_LOS_Recived_power}')
        Indoor_LOS_SNR=DoSNR(1,dBmToWatt(Indoor_LOS_Recived_power),dBmToWatt(N0))
        LOS_SNR=DoSNR(1,dBmToWatt(LOS_Recived_power),dBmToWatt(N0))
        NLOS_SNR=DoSNR(1,dBmToWatt(NLOS_Recived_power),dBmToWatt(N0))
        print(f'LOS_SNR:{LOS_SNR} , NLOS_SNR:{NLOS_SNR},Indoor_LOS_SNR:{Indoor_LOS_SNR}')
        # SNR = DoSNR(1,dBToWatt(-106.41),dBToWatt(N0))
        Indoor_LOS_thr = DoThroughput(transmit_bandwidth,Indoor_LOS_SNR)
        LOS_thr=DoThroughput(transmit_bandwidth,LOS_SNR)
        NLOS_thr=DoThroughput(transmit_bandwidth,NLOS_SNR)
        print(f'LOS_thr : {LOS_thr:g} , NLOS_thr : {NLOS_thr:g}, Indoor_LOS_thr: {Indoor_LOS_thr:g}')
        print('============================')

    
    # NNo
    # NNo