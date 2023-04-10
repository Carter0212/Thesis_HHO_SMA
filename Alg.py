from Base import Root
import numpy as np
import random
import math
class HHO_SMA(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        # super(HHO, self).__init__(function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func


    def run(self):
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
                r8 = np.random.normal(0,1)
                if np.random.normal(0,1) <0.03:
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
                    pop[i,self.ID_POS] = (pop[i,self.ID_POS]*np.random.normal(0,1)).astype(int)

                pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        
        self.convergence=convergence_curve

    
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

    def run(self):
        t = 0
        convergence_curve=np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_Rabbit =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)
        while t < self.iteration:
            original_pop = pop.copy()

            # self.progress(t,self.iteration,status="HHO is running...")
            E1=2*(1-(t/self.iteration)) # factor to show the decreaing energy of rabbit 

            for i in range(self.problem_size):
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
            pop,best_Rabbit = self.update_sorted_population_and_global_best_solution(pop,self.ID_MIN_PROB,best_Rabbit,self.compare_func,self.compare_bool_func,original_pop)   
            convergence_curve[t]=best_Rabbit[self.ID_FIT][2]
            if (t%1==0):
                    print(['At iteration '+ str(t)+ ' the best fitness is '+ str(best_Rabbit[self.ID_FIT])])
            t=t+1
        
        self.convergence=convergence_curve

    
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
                    pop[i][self.ID_POS]  = np.random.uniform(self.lb, self.ub,self.dimension)
                else:
                    p = np.tanh(abs(pop[i][self.ID_FIT][2] - g_best[self.ID_FIT][2]))    # Eq.(2.2)
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
            convergence_curve[t]=g_best[self.ID_FIT][2]
            # if (t%1==0):
            #         print(['At iteration '+ str(t)+ ' the best fitness is '+ str(g_best[self.ID_FIT])])
            t=t+1
        self.convergence=convergence_curve


# algorithm_parameters={'max_num_iteration': None,\
#                                        'population_size':100,\
#                                        'mutation_probability':0.1,\
#                                        'elit_ratio': 0.01,\
#                                        'crossover_probability': 0.5,\
#                                        'parents_portion': 0.3,\
#                                        'crossover_type':'uniform',\
#                                        'max_iteration_without_improv':None},\

class GA(Root):
    def __init__(self,function,dimension,iteration,problem_size,lb,ub,compare_func,compare_bool_func,parents_portion,mutation_prob,crossover_prob):
        Root.__init__(self,function,dimension,iteration,problem_size,lb,ub)
        self.compare_func = compare_func
        self.compare_bool_func = compare_bool_func

        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]" 

        self.parents_portion = parents_portion
        self.par_s=int(self.parents_portion*self.problem_size)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1


        self.mutation_prob = mutation_prob
        assert (self.mutation_prob<=1 and self.mutation_prob>=0), \
        "mutation_probability must be in range [0,1]"

        self.crossover_prob=crossover_prob
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"

        assert (self.elit_ratio <=1 and self.elit_ratio>=0),\
        "elit_ratio must be in range [0,1]" 

        trl=self.pop_s*self.elit_ratio
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)

        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
    
    def run(self):
        t = 0
        convergence_curve=np.zeros(self.iteration)
        pop = [self.create_solution() for _ in range(self.problem_size)]
        pop=np.array(pop)
        pop,best_chromosome =self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB,self.compare_func)

        while t < self.iteration:
            self.progress(t,self.iteration,status="GA is running...")

    def crossover(self,chromosome1,chromosome2):
         
        ofs1=chromosome1.copy()
        ofs2=chromosome2.copy()
        

        if self.c_type=='one_point':
            ran=np.random.randint(0,self.dimension)
            for i in range(0,ran):
                ofs1[i]=chromosome2[i].copy()
                ofs2[i]=chromosome1[i].copy()
  
        if self.c_type=='two_point':
                
            ran1=np.random.randint(0,self.dimension)
            ran2=np.random.randint(ran1,self.dimension)
                
            for i in range(ran1,ran2):
                ofs1[i]=chromosome2[i].copy()
                ofs2[i]=chromosome1[i].copy()
            
        if self.c_type=='uniform':
                
            for i in range(0, self.dimension):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=chromosome2[i].copy()
                    ofs2[i]=chromosome1[i].copy() 
                   
        return np.array([ofs1,ofs2])
    
    def mut(self,chromosome):
        
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                
                chromosome[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        

        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   

               chromosome[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
            
        return x

        
    def show_time(self):
        print(self.executionTime)

    def get_best_individual(self):
        pass