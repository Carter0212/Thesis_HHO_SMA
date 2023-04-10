# -*- coding: utf-8 -*-

import random
import numpy
import math
from solution import solution
import time
import numpy as np
import math
import functools

def compare_Best(news,olds,t):
    if t==0:
        return True
    if news[0]==True and olds[0] == True and news[2]>olds[2]:
        return True
    if news[0]==True and olds[0] == False:
       return True
    if news[0]==False and olds[0] == False and news[3] < olds[3]:
        return True
    return False



def LCSDP(t,lb,ub,C,X_hho,SearchAgents_no,dim):
    if t==0:
        C=random.random()
        C_1=C
    else:
        C_1 = random.random()*C*(1-C)
    
    for i in range(0,SearchAgents_no):
        choice_dim = np.random.randint(0,dim)
        X_hho[i,choice_dim] = lb + C*(ub-lb)
    
    return X_hho,C_1

def Non_Domainated_Sort(merge_X,SearchAgents_no,objf):
    Z = numpy.asarray([])
    for i in range(0,3*SearchAgents_no):
        fitness=objf(merge_X[i,:])
        Z=np.extend(Z,fitness)
        print(Z)
    print(np.shape(Z))
    exit(1)
    


        



def HHO(objf,lb,ub,dim,SearchAgents_no,Max_iter):

    # dim=30
    # SearchAgents_no=50 
    # lb=-100
    # ub=100
    # Max_iter=500
    
    # initialize the location and Energy of the rabbit
    Rabbit_Location=numpy.zeros(dim)
    Rabbit_Energy=float("-inf")  #change this to -inf for maximization problems
    
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)
         
    #Initialize the locations of Harris' hawks
    X=numpy.asarray([x*(ub-lb)+lb for x in numpy.random.uniform(0,1,(SearchAgents_no, dim))])
    
    #Initialize convergence
    convergence_curve=numpy.zeros(Max_iter)
    
    ############################
    s=solution()

    print("HHO is now tackling  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    C=0
    t=0  # Loop counter
    
    # Main loop
    while t<Max_iter:
        for i in range(0,SearchAgents_no):
            
            # Check boundries
                      
            X[i,:]=numpy.clip(X[i,:], lb, ub)
            
            # fitness of locations
            
            fitness=objf(X[i,:])
            # print(fitness)
            
            # Update the location of Rabbit
            if compare_Best(fitness,Rabbit_Energy,t): # Change this to > for maximization problem
                Rabbit_Energy=fitness 
                Rabbit_Location=X[i,:].copy() 
        
        E1=2*(1-(t/Max_iter)) # factor to show the decreaing energy of rabbit    
        X_orngal = X.copy()
        # Update the location of Harris' hawks 
        for i in range(0,SearchAgents_no):

            E0=2*random.random()-1  # -1<E0<1
            Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy)>=1:
                #Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no*random.random())
                X_rand = X[rand_Hawk_index, :]
                if q<0.5:
                    # perch based on other family members
                    X[i,:]=X_rand-random.random()*abs(X_rand-2*random.random()*X[i,:])

                elif q>=0.5:
                    #perch on a random tall tree (random site inside group's home range)
                    X[i,:]=(Rabbit_Location - X.mean(0))-random.random()*((ub-lb)*random.random()+lb)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy)<1:
                #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                #phase 1: ----- surprise pounce (seven kills) ----------
                #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r=random.random() # probablity of each event
                
                if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                    X[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X[i,:])

                if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                    Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                    X[i,:]=(Rabbit_Location-X[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                
                #phase 2: --------performing team rapid dives (leapfrog movements)----------

                if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                    #rabbit try to escape by many zigzag deceptive motions
                    Jump_strength=2*(1-random.random())
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                    X1 = numpy.clip(X1, lb, ub)

                    if compare_Best(objf(X1),fitness,t): # improved move?
                        X[i,:] = X1.copy()
                    else: # hawks perform levy-based short rapid dives around the rabbit
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                        X2 = numpy.clip(X2, lb, ub)
                        if compare_Best(objf(X2),fitness,t):
                            X[i,:] = X2.copy()
                if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                     Jump_strength=2*(1-random.random())
                     X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))
                     X1 = numpy.clip(X1, lb, ub)
                     
                     if compare_Best(objf(X1),fitness,t): # improved move?
                        X[i,:] = X1.copy()
                     else: # Perform levy-based short rapid dives around the rabbit
                         X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                         X2 = numpy.clip(X2, lb, ub)
                         if compare_Best(objf(X2),fitness,t):
                            X[i,:] = X2.copy()


        X_LCSDP,C = LCSDP(t,lb,ub,C,X,SearchAgents_no,dim)
        Non_Domainated_Sort(np.concatenate([X_orngal,X_LCSDP,X]),SearchAgents_no,objf)

        convergence_curve[t]=Rabbit_Energy[2]
        if (t%1==0):
                print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Rabbit_Energy)])
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="HHO"   
    s.objfname=objf.__name__
    s.best=Rabbit_Energy 
    s.bestIndividual = Rabbit_Location
    s.show_time()
    s.pri()
    
    
    
    
    return s

def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step

def Rosenbrock(x):
    # return max(x)
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

if __name__ == "__main__":
    # lb = [-5, -5]  # 下限
    # ub = [5, 5]  # 上限
    # dim = 2  # 維度
    # SearchAgents_no = 50  # 搜尋代理人數
    # Max_iter = 1000  # 迭代次數
    dim=10000
    SearchAgents_no=100
    lb=0
    ub=1
    Max_iter=10
    HHO(Rosenbrock, lb, ub, dim, SearchAgents_no, Max_iter)
    

