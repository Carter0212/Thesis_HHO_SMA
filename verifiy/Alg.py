from Base import Root
import numpy as np
import random
import math
import time

class HHO_SMA_ch(Root):

    ID_MESK = 5
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,ue_numbers,base_numbers):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.ue_numbers = ue_numbers
        self.base_numbers = base_numbers


    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            
            # self.progress(t,self.iteration,status="HHO is running...")
            E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            p = 1-(t/self.iteration)
            for i in range(self.problem_size):
                if not pop[i][self.ID_FIT][0]:
                    ALL_MASK=pop[i,self.ID_FIT][self.ID_MESK]
                    Over_Power_mask = ALL_MASK[3]
                    rate_inadeuate_mask = ALL_MASK[2]
                    Zero_mask_connect_UE = ALL_MASK[1]
                    Zero_mask_connect_BS = ALL_MASK[0]
                    if(i==0):
                        print(pop[i,self.ID_POS])
                        print(self.dimension)
                        print(rate_inadeuate_mask,Over_Power_mask)
                        
                    for dim in range(self.dimension):
                        rate_check=np.isin(rate_inadeuate_mask,abs(dim-self.dimension//2)//self.base_numbers)
                        Zero_mask_connect_UE_Check = np.isin(Zero_mask_connect_UE,abs(dim-self.dimension//2)//self.base_numbers)
                        Zero_mask_connect_BS_Check = np.isin(Zero_mask_connect_BS,abs(dim-self.dimension//2)%self.base_numbers)
                        Power_check=np.isin(Over_Power_mask,abs(dim-self.dimension//2)%self.base_numbers)
                        test_mask=[]
                        change_connectORpower = random.choice([True, False])
                        
                        if dim < self.dimension//2:
                            if True in Zero_mask_connect_UE_Check:  
                                if pop[i,self.ID_POS][dim] < 500:
                                    pop[i,self.ID_POS][dim]*=0.8
                                    test_mask.append(dim)
                            if True in Zero_mask_connect_BS_Check:  
                                if pop[i,self.ID_POS][dim] < 500:
                                    pop[i,self.ID_POS][dim]*=0.8
                                    test_mask.append(dim)
                            if change_connectORpower and rate_check and pop[i,self.ID_POS][dim] <500:
                                pop[i,self.ID_POS][dim]*=1.2

                            
                        else:
                            # check Power
                            rate_check=np.isin(rate_inadeuate_mask,(dim//self.base_numbers))
                            Zero_mask_connect_UE_Check = np.isin(Zero_mask_connect_UE,dim//self.base_numbers)
                            if True in rate_check and Power_check in False:
                                if i == 0:
                                    print(rate_check,dim)   
                                if pop[i,self.ID_POS][dim] < 500:
                                    if i == 0:
                                        print(Power_check,dim)
                                    pop[i,self.ID_POS][dim]*=1.2
                                    test_mask.append(dim)
                            if True in Power_check and pop[i,self.ID_POS][dim] < 600 and not change_connectORpower:
                                pop[i,self.ID_POS][dim]*=1.2
                                test_mask.append(dim)
                    if(i==0):
                        print(pop[i,self.ID_POS])
                        
                        print(test_mask)
                    
                    
                    
                r8 = np.random.normal(0,1)
                j = 1-(i/self.problem_size)
                if j <= 0.3:
                    a= np.random.randint(self.lb, self.ub, size=self.dimension)
                    pop[i,self.ID_POS] = a
                elif r8 < p:
                    E0=2*random.random()-1  # -1<E0<1
                    Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                    # -------- Exploration phase Eq. (1) in paper -------------------#

                    if abs(Escaping_Energy)>=1:
                        q = random.random()
                        rand_Hawk_index = math.floor(self.problem_size*random.random())
                        pop_rand = pop[rand_Hawk_index,0]
                        if q<0.5:
                        # perch based on other family members
                            pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                            
                        elif q>=0.5:
                            #perch on a random tall tree (random site inside group's home range)
                            pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                    elif abs(Escaping_Energy)<1:
                        #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                        #phase 1: ----- surprise pounce (seven kills) ----------
                        #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                        r=random.random() # probablity of each event
                        if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                            pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        
                        if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                            Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                            pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])

                        
                        #phase 2: --------performing team rapid dives (leapfrog movements)----------

                        if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                            #rabbit try to escape by many zigzag deceptive motions
                            Jump_strength=2*(1-random.random())
                            X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                            X1 = np.clip(X1, self.lb, self.ub)
                            X1_fitness = self.obj_func(X1)
                            if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X1.copy(),X1_fitness)
                            else: # hawks perform levy-based short rapid dives around the rabbit
                                X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                                X2 = np.clip(X2, self.lb, self.ub)
                                X2_fitness = self.obj_func(X2)
                                if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                    pop[i] = (X2.copy(),X2_fitness) 
                        
                        if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                            Jump_strength=2*(1-random.random())
                            X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                            X1 = np.clip(X1, self.lb, self.ub)
                            X1_fitness = self.obj_func(X1)
                            if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X1.copy(),X1_fitness)
                            else: # Perform levy-based short rapid dives around the rabbit
                                X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                                X2 = np.clip(X2, self.lb, self.ub)
                                X2_fitness = self.obj_func(X2)
                                if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                    pop[i] = (X2.copy(),X2_fitness)
                elif r8 >= p:
                    pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_POS] = self.amend_position(pop[i,self.ID_POS])
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.convergence=convergence_curve
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step
    
    def Weight(self,pop,problem):
        pass
        # POS=np.reshape(pop[problem,self.ID_POS],(2,base_numbers,ue_numbers)).copy()
        # if pop[problem,self.ID_FIT][0]:
        #     pass
        # else:
        #     check_conectional = (POS[0] >= 500)
        #     if not pop[problem,self.ID_FIT][1][0]:
        #         pass
        #     if not pop[problem,self.ID_FIT][1][1]:
        #         pass
        #     if not pop[problem,self.ID_FIT][1][2]:
        #         Zero_mask_connect_BS=np.where((np.sum(check_conectional,axis=1) <= 0))
        #     if not pop[problem,self.ID_FIT][1][3]:
        #         Zero_mask_connect_UE=np.where((np.sum(check_conectional,axis=1) <= 0))
        #     if not pop[problem,self.ID_FIT][1][4]:
        #         np.sum(Base_ue_ThroughtpusTable,axis=0) >= Min_Rate
        #     if not pop[problem,self.ID_FIT][1][5]:
        #         # happen Power Over Maximum constraint
        #         # check Which
        #         BS_power=np.sum(POS[1]*check_conectional,axis=1)
        #         Power_mask=(BS_power > 1000)
        #         whichBSOverPower=np.where(Power_mask)
        #         POS[1,whichBSOverPower,:]=

class Random(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = 4


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            # self.progress(t,self.iteration,status="Random is running...")
            original_pop = pop.copy()
            # original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            
            for i in range(self.problem_size):
                pop[i,self.ID_POS] = np.asarray([int (x*(self.ub-self.lb)+self.lb) for x in np.random.uniform(0,1,self.dimension)])
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

class MaxPower(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = 4


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            
            for i in range(self.problem_size):
                pop[i,self.ID_POS] = np.asarray([int (x*(self.ub-self.lb)+self.lb) for x in np.random.uniform(0,1,self.dimension)])
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

class HHO_DE_Chaos(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = 4
        self.CR = 0.7


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            chaos_pop=self.chaos(original_pop.copy())
            original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            # self.progress(t,self.iteration,status="HHO is running...")
            # E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            E1 = np.log(np.power(t/self.iteration,1/3))
            p = 1-(t/self.iteration)
            for i in range(self.problem_size):
                r8 = np.random.normal(0,1)
                r7 = np.random.normal(0,1)
                # if np.random.normal(0,1) <0.03:
                #     a= np.random.randint(self.lb, self.ub, size=self.dimension)
                #     pop[i,self.ID_POS] = a
                # elif r8 < p:
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        X3 = best_Rabbit[self.ID_POS]+r7*(pop[:,self.ID_POS].mean(0)-pop[i,self.ID_POS])
                        X3_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X3_fitness=self.obj_func(X3)
                        X3_1_fitness=self.obj_func(X3_1)
                        if self.compare_bool_func(X3_fitness,X3_1_fitness):
                            pop[i]=(X3.copy(),X3_fitness)
                        else:
                            pop[i]=(X3_1.copy(),X3_1_fitness)
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        X4=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_fitness=self.obj_func(X4)
                        X4_1_fitness=self.obj_func(X4_1)
                        if self.compare_bool_func(X4_fitness,X4_1_fitness):
                            pop[i]=(X4.copy(),X4_fitness)
                        else:
                            pop[i]=(X4_1.copy(),X4_1_fitness)
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                # elif r8 >= p:
                #     pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
                self.DE_Strategy(pop,i)
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT]
            
            # constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    def chaos(self,original_pop):
        chaos = random.random()

        for ch in range(self.problem_size):
            chaos = self.Mu * chaos * (1-chaos)
            buffer=original_pop[ch,self.ID_POS]*chaos
            original_pop[ch,self.ID_POS]= buffer
            original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])    
        return original_pop

    def DE_Strategy(self,pop,i):
        choice_three_pop=random.sample(range(0,self.problem_size),3)
        while i in choice_three_pop:
            choice_three_pop=random.sample(range(0,self.problem_size),3)
        V_POS=pop[choice_three_pop[0]][self.ID_POS] + (np.random.uniform(0.2,0.8)*(pop[choice_three_pop[1]][self.ID_POS]-pop[choice_three_pop[2]][self.ID_POS]))
        U_POS=np.zeros(self.dimension)

        CR_compare =np.random.uniform(size=self.dimension) <= self.CR
        U_POS = V_POS*CR_compare + pop[i][self.ID_POS]*(~CR_compare)


        ##Crossover operation
        # for dim in range(self.dimension):
        #     if np.random.random() <= self.CR:
        #         U_POS[dim] = V_POS[dim]
        #     else:
        #         U_POS[dim] = pop[i][self.ID_POS][dim]

        ## Selection operation
        U_FIT=self.get_fitness(U_POS)
        if self.compare_bool_func(U_FIT,pop[i][self.ID_FIT]):
            pop[i]=(U_POS.copy(),U_FIT)

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class HHO_SMA_Chaos(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.Mu = 4


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            chaos_pop=self.chaos(original_pop.copy())
            original_pop,best_Rabbit=self.update_sorted_population_and_global_best_solution(original_pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,chaos_pop)
            # self.progress(t,self.iteration,status="HHO is running...")
            # E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            E1 = np.log(np.power(t/self.iteration,1/3))
            p = 1-(t/self.iteration)
            for i in range(self.problem_size):
                r8 = np.random.normal(0,1)
                r7 = np.random.normal(0,1)
                # if np.random.normal(0,1) <0.03:
                #     a= np.random.randint(self.lb, self.ub, size=self.dimension)
                #     pop[i,self.ID_POS] = a
                # elif r8 < p:
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        X3 = best_Rabbit[self.ID_POS]+r7*(pop[:,self.ID_POS].mean(0)-pop[i,self.ID_POS])
                        X3_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X3_fitness=self.obj_func(X3)
                        X3_1_fitness=self.obj_func(X3_1)
                        if self.compare_bool_func(X3_fitness,X3_1_fitness):
                            pop[i]=(X3.copy(),X3_fitness)
                        else:
                            pop[i]=(X3_1.copy(),X3_1_fitness)
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        X4=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_fitness=self.obj_func(X4)
                        X4_1_fitness=self.obj_func(X4_1)
                        if self.compare_bool_func(X4_fitness,X4_1_fitness):
                            pop[i]=(X4.copy(),X4_fitness)
                        else:
                            pop[i]=(X4_1.copy(),X4_1_fitness)
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                # elif r8 >= p:
                #     pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    def chaos(self,original_pop):
        chaos = random.random()

        for ch in range(self.problem_size):
            chaos = self.Mu * chaos * (1-chaos)
            buffer=original_pop[ch,self.ID_POS]*chaos
            original_pop[ch,self.ID_POS]= buffer
            original_pop[ch,self.ID_FIT] = self.get_fitness(original_pop[ch,self.ID_POS])    
        return original_pop

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class HHO_SMA_DE(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.CR=0.2

    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            # self.progress(t,self.iteration,status="HHO is running...")
            # E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            E1 = np.log(np.power(t/self.iteration,1/3))
            p = 1-(t/self.iteration)
            for i in range(self.problem_size):
                r8 = np.random.normal(0,1)
                r7 = np.random.normal(0,1)
                # if np.random.normal(0,1) <0.03:
                #     a= np.random.randint(self.lb, self.ub, size=self.dimension)
                #     pop[i,self.ID_POS] = a
                # elif r8 < p:
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        X3 = best_Rabbit[self.ID_POS]+r7*(pop[:,self.ID_POS].mean(0)-pop[i,self.ID_POS])
                        X3_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X3_fitness=self.obj_func(X3)
                        X3_1_fitness=self.obj_func(X3_1)
                        if self.compare_bool_func(X3_fitness,X3_1_fitness):
                            pop[i]=(X3.copy(),X3_fitness)
                        else:
                            pop[i]=(X3_1.copy(),X3_1_fitness)
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        X4=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_fitness=self.obj_func(X4)
                        X4_1_fitness=self.obj_func(X4_1)
                        if self.compare_bool_func(X4_fitness,X4_1_fitness):
                            pop[i]=(X4.copy(),X4_fitness)
                        else:
                            pop[i]=(X4_1.copy(),X4_1_fitness)
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                # elif r8 >= p:
                #     pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
                self.DE_Strategy(pop,i)
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step
    
    
    def DE_Strategy(self,pop,i):
        choice_three_pop=random.sample(range(0,self.problem_size),3)
        while i in choice_three_pop:
            choice_three_pop=random.sample(range(0,self.problem_size),3)
        V_POS=pop[choice_three_pop[0]][self.ID_POS] + (np.random.uniform(0.2,0.8)*(pop[choice_three_pop[1]][self.ID_POS]-pop[choice_three_pop[2]][self.ID_POS]))
        U_POS=np.zeros(self.dimension)

        CR_compare =np.random.uniform(size=self.dimension) <= self.CR
        U_POS = V_POS*CR_compare + pop[i][self.ID_POS]*(~CR_compare)


        ##Crossover operation
        # for dim in range(self.dimension):
        #     if np.random.random() <= self.CR:
        #         U_POS[dim] = V_POS[dim]
        #     else:
        #         U_POS[dim] = pop[i][self.ID_POS][dim]

        ## Selection operation
        U_FIT=self.get_fitness(U_POS)
        if self.compare_bool_func(U_FIT,pop[i][self.ID_FIT]):
            pop[i]=(U_POS.copy(),U_FIT)


class HHO_SMA(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func


    def run(self):
        timerStart=time.time() 
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()
            # self.progress(t,self.iteration,status="HHO is running...")
            # E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 
            E1 = np.log(np.power(t/self.iteration,1/3))
            p = 1-(t/self.iteration)
            for i in range(self.problem_size):
                r8 = np.random.normal(0,1)
                r7 = np.random.normal(0,1)
                # if np.random.normal(0,1) <0.03:
                #     a= np.random.randint(self.lb, self.ub, size=self.dimension)
                #     pop[i,self.ID_POS] = a
                # elif r8 < p:
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#

                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        X3 = best_Rabbit[self.ID_POS]+r7*(pop[:,self.ID_POS].mean(0)-pop[i,self.ID_POS])
                        X3_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X3_fitness=self.obj_func(X3)
                        X3_1_fitness=self.obj_func(X3_1)
                        if self.compare_bool_func(X3_fitness,X3_1_fitness):
                            pop[i]=(X3.copy(),X3_fitness)
                        else:
                            pop[i]=(X3_1.copy(),X3_1_fitness)
                    
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        X4=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_1=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X4_fitness=self.obj_func(X4)
                        X4_1_fitness=self.obj_func(X4_1)
                        if self.compare_bool_func(X4_fitness,X4_1_fitness):
                            pop[i]=(X4.copy(),X4_fitness)
                        else:
                            pop[i]=(X4_1.copy(),X4_1_fitness)
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness) 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                # elif r8 >= p:
                #     pop[i,self.ID_POS] = pop[i,self.ID_POS]*np.random.normal(0,1,size=self.dimension)
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT][2] 
        self.bestIndividual = best_Rabbit[self.ID_POS]

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class HHO(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func

    def check_fuc(self,value,printWord):
        if np.any(value<0):
            print(printWord)
            print(value)
            exit(1)
        else:
            pass


    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            # original_pop = pop.copy()

            # self.progress(t,self.iteration,status="HHO is running...")
            E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 

            for i in range(self.problem_size):
                
                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper


                # -------- Exploration phase Eq. (1) in paper -------------------#
                pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func)
                if abs(Escaping_Energy)>=1:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.problem_size*random.random())
                    pop_rand = pop[rand_Hawk_index,0]
                    if q<0.5:
                    # perch based on other family members
                        pop[i,self.ID_POS]=pop_rand-random.random()*abs(pop_rand-2*random.random()*pop[i,self.ID_POS])
                        # self.check_fuc(pop[i,self.ID_POS],'E>=1,q<0.5')
                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS] - pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)
                        # self.check_fuc(pop[i,self.ID_POS],'E>=1,q>=0.5')
                elif abs(Escaping_Energy)<1:
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS])-Escaping_Energy*abs(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        # self.check_fuc(pop[i,self.ID_POS],'E<0.5,r>=0.5')
                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        pop[i,self.ID_POS]=(best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        # self.check_fuc(pop[i,self.ID_POS],'E>=0.5,r>=0.5')
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[i,self.ID_POS])+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                                pop[i] = (X2.copy(),X2_fitness)
                        # self.check_fuc(pop[i,self.ID_POS],'E>=0.5,q<0.5') 
                    
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        Jump_strength=2*(1-random.random())
                        X1=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))
                        X1 = np.clip(X1, self.lb, self.ub)
                        X1_fitness = self.obj_func(X1)
                        if self.compare_bool_func(X1_fitness,best_Rabbit[self.ID_FIT]): # improved move?
                            pop[i] = (X1.copy(),X1_fitness)
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=best_Rabbit[self.ID_POS]-Escaping_Energy*abs(Jump_strength*best_Rabbit[self.ID_POS]-pop[:,self.ID_POS].mean(0))+np.multiply(np.random.randn(self.dimension),self.Levy(self.dimension))
                            X2 = np.clip(X2, self.lb, self.ub)
                            X2_fitness = self.obj_func(X2)
                            if self.compare_bool_func(X2_fitness,best_Rabbit[self.ID_FIT]):
                                pop[i] = (X2.copy(),X2_fitness)
                        # self.check_fuc(pop[i,self.ID_POS],'E>0.5,q<0.5')
                pop[i,self.ID_POS] = self.amend_position(pop[i,self.ID_POS])
                pop[i,self.ID_FIT]=self.get_fitness(pop[i,self.ID_POS])
               
            convergence_curve[t]=best_Rabbit[self.ID_FIT]
            # constrained_violation_curve[t] = best_Rabbit[self.ID_FIT][3]
            # self.posRecord.append(pop[:,self.ID_POS])
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_Rabbit[self.ID_FIT]
        self.bestIndividual = best_Rabbit[self.ID_POS]
        
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve

    
    def Levy(self,dim):
        beta=1.5
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
        u= 0.01*np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v),(1/beta))
        step = np.divide(u,zz)
        return step

class BaseSMA(Root):
    """
    Modified version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
    Notes:
            + Selected 2 unique and random solution to create new solution (not to create variable) --> remove third loop in original version
            + Check bound and update fitness after each individual move instead of after the whole population move in the original version
    """

    ID_WEI = 2

    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,z=0.03):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.z = z

    def create_solution(self, minmax=0):
        # if not isinstance(self.lb, list):
        #     self.lb = [self.lb for _ in range(self.dimension)]
        #     self.ub = [self.ub for _ in range(self.dimension)]
        # self.lb = np.asarray(self.lb)
        # self.ub = np.asarray(self.ub)
        pos = np.asarray([x*(self.ub-self.lb)+self.lb for x in np.random.uniform(0,1,self.dimension)])
        fit = self.get_fitness(pos)
        weight = np.zeros(self.dimension)
        return (pos, fit,weight)
    
    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        
        
        pop,g_best =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            self.progress(t,self.iteration,status="BaseSMA is running...")
            # print(pop[:,self.ID_FIT])
            s = pop[0][self.ID_FIT][2] - pop[-1][self.ID_FIT][2] + self.EPSILON
            
             # calculate the fitness weight of each slime mold
            for i in range(0, self.problem_size):
                # Eq.(2.5)
                if i <= int(self.problem_size / 2):
                    pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT][2] - pop[i][self.ID_FIT][2])) / s + 1)
                else:
                    pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT][2] - pop[i][self.ID_FIT][2])) / s + 1)

            a = np.arctanh(-((t + 1) / self.iteration) + 1)                        # Eq.(2.4)
            b = 1 - (t + 1) / self.iteration

            # Update the Position of search agents
            for i in range(0, self.problem_size):
                
                if np.random.uniform() < self.z:  # Eq.(2.7)
                    pos_new = np.random.uniform(self.lb, self.ub,self.dimension)
                else:
                    p = np.tanh(abs(pop[i][self.ID_FIT][2] - g_best[self.ID_FIT][2]))    # Eq.(2.2)
                    vb = np.random.uniform(-a, a, self.dimension)                      # Eq.(2.3)
                    vc = np.random.uniform(-b, b, self.dimension)

                    # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                    id_a, id_b = np.random.choice(list(set(range(0, self.problem_size)) - {i}), 2, replace=False)
                    pos_1 = g_best[self.ID_POS] + vb * (pop[i][self.ID_WEI] * pop[id_a][self.ID_POS] - pop[id_b][self.ID_POS])
                    pos_2 = vc * pop[i][self.ID_POS]
                    pos_new = np.where(np.random.uniform(0, 1, self.dimension) < p, pos_1, pos_2)

                # Check bound and re-calculate fitness after each individual move
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_fitness(pos_new)
                pop[i][self.ID_POS] = pos_new
                pop[i][self.ID_FIT] = fit_new
                
            
            # Sorted population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,g_best,self.compare_func,self.compare_bool_func)   
            convergence_curve[t]=g_best[self.ID_FIT][2]
            # if (t%1==0):
            #         print(['At iteration '+ str(t)+ ' the best fitness is '+ str(g_best[self.ID_FIT])])
            t=t+1
        self.convergence=convergence_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=g_best[self.ID_FIT][2] 
        self.bestIndividual = g_best[self.ID_POS]
        
import warnings

class OriginalSMA(Root):
    """
        The original version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
        Link:
            https://doi.org/10.1016/j.future.2020.03.055
    """

    ID_WEI = 2

    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,z=0.03):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.z = z

    def create_solution(self, minmax=0):
        # if not isinstance(self.lb, list):
        #     self.lb = [self.lb for _ in range(self.dimension)]
        #     self.ub = [self.ub for _ in range(self.dimension)]
        # self.lb = np.asarray(self.lb)
        # self.ub = np.asarray(self.ub)
        pos = np.asarray([x*(self.ub-self.lb)+self.lb for x in np.random.uniform(0,1,self.dimension)])
        fit = self.get_fitness(pos)
        weight = np.zeros(self.dimension)
        return (pos, fit,weight)
    
    def run(self):
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        
        
        pop,g_best =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        
        
        while t < self.iteration:
            # self.progress(t,self.iteration,status="BaseSMA is running...")
            s = abs(pop[0][self.ID_FIT] - pop[-1][self.ID_FIT] + self.EPSILON)
            
             # calculate the fitness weight of each slime mold

            for i in range(0, self.problem_size):
                # Eq.(2.5)
                
                    if i <= int(self.problem_size / 2):

                        # print(f'abs:{abs((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]))}')
                        # print(f's:{s}')
                        pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT] - pop[i][self.ID_FIT])) / s + 1)
                    else:
                        # print(f'abs2:{abs((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]))}')
                        # print(f's2:{s}')
                        pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.dimension) * np.log10(abs((pop[0][self.ID_FIT] - pop[i][self.ID_FIT])) / s + 1)
                # if 
                # print("pop[0][self.ID_FIT]:", pop[0][self.ID_FIT])
                # print("pop[i][self.ID_FIT]:", pop[i][self.ID_FIT])
                # print("s:", s)
                # print('warning')
                # exit(1)

            a = np.arctanh(-((t + 1) / self.iteration) + 1)                        # Eq.(2.4)
            b = 1 - (t + 1) / self.iteration

            # Update the Position of search agents
            for i in range(0, self.problem_size):
                
                if np.random.uniform() < self.z:  # Eq.(2.7)
                    pop[i][self.ID_POS]  = np.random.uniform(self.lb, self.ub,self.dimension)
                else:
                    p = np.tanh(abs(pop[i][self.ID_FIT] - g_best[self.ID_FIT]))    # Eq.(2.2)
                    vb = np.random.uniform(-a, a, self.dimension)                      # Eq.(2.3)
                    vc = np.random.uniform(-b, b, self.dimension)
                    for j in range(0, self.dimension):
                    # two positions randomly selected from population
                        id_a, id_b = np.random.choice(list(set(range(0, self.problem_size)) - {i}), 2, replace=False)

                        if np.random.uniform() < p:
                            pop[i][self.ID_POS][j] = g_best[self.ID_POS][j] + vb[j] * (
                                        pop[i][self.ID_WEI][j] * pop[id_a][self.ID_POS][j] - pop[id_b][self.ID_POS][j])
                        else:
                            pop[i][self.ID_POS][j] = vc[j] * pop[i][self.ID_POS][j]
                    pos_1 = g_best[self.ID_POS] + vb * (pop[i][self.ID_WEI] * pop[id_a][self.ID_POS] - pop[id_b][self.ID_POS])
                    pos_2 = vc * pop[i][self.ID_POS]
                    pos_new = np.where(np.random.uniform(0, 1, self.dimension) < p, pos_1, pos_2)

                # Check bound and re-calculate fitness after each individual move
            for i in range(0, self.problem_size):
                pos_new = self.amend_position(pop[i][self.ID_POS])
                fit_new = self.get_fitness(pos_new)
                pop[i][self.ID_POS] = pos_new
                pop[i][self.ID_FIT] = fit_new
            
            
            # Sorted population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,g_best,self.compare_func,self.compare_bool_func)   
            
            convergence_curve[t]=g_best[self.ID_FIT]
            # constrained_violation_curve[t] = g_best[self.ID_FIT][3]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(g_best[self.ID_FIT])])
            t=t+1
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=g_best[self.ID_FIT]
        self.bestIndividual = g_best[self.ID_POS]

class GA(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,parents_portion,mutation_prob,crossover_prob,elit_ratio,cross_type):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func
        self.elit_ratio = elit_ratio
        self.parents_portion = parents_portion
        assert (self.parents_portion<=1\
                and self.parents_portion>=0),\
        "parents_portion must be in range [0,1]" 
        self.cross_type = cross_type
        self.parents_portion = parents_portion
        self.par_s=int(self.parents_portion*self.problem_size)
        trl=self.problem_size-self.par_s
        if trl % 2 != 0:
            self.par_s+=1


        self.mutation_prob = mutation_prob
        assert (self.mutation_prob<=1 and self.mutation_prob>=0), \
        "mutation_probability must be in range [0,1]"

        self.crossover_prob=crossover_prob
        assert (self.crossover_prob<=1 and self.crossover_prob>=0), \
        "mutation_probability must be in range [0,1]"

        assert (self.elit_ratio <=1 and self.elit_ratio>=0),\
        "elit_ratio must be in range [0,1]" 

        trl=self.problem_size*self.elit_ratio
        if trl<1 and self.elit_ratio>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)

        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
    
    def run(self):
        ## use Roulette Wheel & elit 
        timerStart=time.time() 
        self.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        
        t = 0
        convergence_curve=np.zeros(self.iteration)
        constrained_violation_curve = np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_chromosome =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        
        while t < self.iteration:
            # self.progress(t,self.iteration,status="GA is running...")

            # Normalizing objective function
            normal_obj = np.zeros(self.problem_size)
            for index,i in enumerate(pop[:,self.ID_FIT].copy()):
                normal_obj[index] = i
            # normal_obj = np.zeros(self.problem_size)
            # normal_obj=pop[:,self.ID_FIT].copy()
            # print(np.shape(normal_obj))
            maxnorm = np.amax(normal_obj)
            # print(normal_obj)
            # print('======================')
            # print(maxnorm)
            normal_obj = maxnorm - normal_obj +1

            #############################################################        
            # Calculate probability
            sum_normobj=np.sum(normal_obj)
            prob=np.zeros(self.problem_size)
            prob=normal_obj/sum_normobj
            
            cumprob=np.cumsum(prob)
            par = np.zeros((self.par_s,2),dtype=pop.dtype)
            # Select elite individuals
            # print(pop[0,1].dtype)
            par[0:self.num_elit] = pop[0:self.num_elit].copy()

            # Select non-elite individuals using roulette wheel selection
            index=np.searchsorted(cumprob,np.random.random(self.par_s-self.num_elit))
            par[self.num_elit:self.par_s]=pop[index].copy()

            ef_par_list = (np.random.random(self.par_s)<=self.crossover_prob)
            par_count = ef_par_list.sum()


            elite_par=par[ef_par_list].copy()

            ## New generation

            pop[:self.par_s] = par[:self.par_s].copy()
            
            for k in range(self.par_s,self.problem_size,2):
                r1 = np.random.randint(0,par_count)
                r2 = np.random.randint(0,par_count)
                pvar1 = elite_par[r1].copy()
                pvar2 = elite_par[r2].copy()

                ch1,ch2 = self.crossover(pvar1,pvar2)

                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)
                pop[k] = ch1.copy()
                pop[k+1] = ch2.copy()
                pop[k,self.ID_FIT]=self.get_fitness(pop[k,self.ID_POS])
                pop[k+1,self.ID_FIT]=self.get_fitness(pop[k+1,self.ID_POS])
            
            pop,best_chromosome = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_chromosome,self.compare_func,self.compare_bool_func)
            convergence_curve[t]=best_chromosome[self.ID_FIT]
            # constrained_violation_curve[t] = best_chromosome[self.ID_FIT][3]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_chromosome[self.ID_FIT])])
            t=t+1
        self.convergence=convergence_curve
        self.constrained_violation_curve = constrained_violation_curve
        timerEnd=time.time()  
        self.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime=timerEnd-timerStart
        self.best=best_chromosome[self.ID_FIT] 
        self.bestIndividual = best_chromosome[self.ID_POS]
        


    def  crossover(self,chromosome1,chromosome2):
         
        ofs1=chromosome1.copy()
        ofs2=chromosome2.copy()
        

        if self.cross_type=='one_point':
            ran=np.random.randint(0,self.dimension)
            for i in range(0,ran):
                ofs1[self.ID_POS][i]=chromosome2[self.ID_POS][i].copy()
                ofs2[self.ID_POS][i]=chromosome1[self.ID_POS][i].copy()
  
        if self.cross_type=='two_point':
                
            ran1=np.random.randint(0,self.dimension)
            ran2=np.random.randint(ran1,self.dimension)
                
            for i in range(ran1,ran2):
                ofs1[self.ID_POS,i]=chromosome2[self.ID_POS,i].copy()
                ofs2[self.ID_POS,i]=chromosome1[self.ID_POS,i].copy()
            
        if self.cross_type=='uniform':
                
            for i in range(0, self.dimension):
                
                ran=np.random.random()
                if ran <0.5:
                    ofs1[self.ID_POS][i]=chromosome2[self.ID_POS][i].copy()
                    ofs2[self.ID_POS][i]=chromosome1[self.ID_POS][i].copy() 
                   
        return ofs1,ofs2
    
    def mut(self,chromosome):
        
        # for i in self.integers[0]:
        #     ran=np.random.random()
        #     if ran < self.mutation_prob:
                
        #         chromosome[i]=self.lb+np.random.random()*(self.ub-self.lb)    
                    
        

        for i in range(self.dimension):            
            ran=np.random.random()
            if ran < self.mutation_prob:   
                chromosome[self.ID_POS][i]=self.lb+np.random.random()*(self.ub-self.lb)    
            
        return chromosome
    
    def mutmidle(self, x, p1, p2):
        # for i in self.integers[0]:
        #     ran=np.random.random()
        #     if ran < self.mutation_prob:
        #         if p1[i]<p2[i]:
        #             x[i]=np.random.randint(p1[i],p2[i])
        #         elif p1[i]>p2[i]:
        #             x[i]=np.random.randint(p2[i],p1[i])
        #         else:
        #             x[i]=np.random.randint(self.var_bound[i][0],\
        #          self.var_bound[i][1]+1)
                        
        for i in range(self.dimension):                
            ran=np.random.random()
            if ran < self.mutation_prob:   
                if p1[self.ID_POS][i]<p2[self.ID_POS][i]:
                    x[self.ID_POS][i]=p1[self.ID_POS][i]+np.random.random()*(p2[self.ID_POS][i]-p1[self.ID_POS][i])  
                elif p1[self.ID_POS][i]>p2[self.ID_POS][i]:
                    x[self.ID_POS][i]=p2[self.ID_POS][i]+np.random.random()*(p1[self.ID_POS][i]-p2[self.ID_POS][i])
                else:
                    x[self.ID_POS][i]=self.lb+np.random.random()*(self.ub-self.lb)   
        return x

        
    def show_time(self):
        print(self.executionTime)

    def get_best_individual(self):
        pass