#!/usr/bin/env python


# Main paper (Please refer to the main paper):
# Slime Mould Algorithm: A New Method for Stochastic Optimization




from numpy.random import uniform, choice
import numpy
from numpy import abs, zeros, log10, where, arctanh, tanh,clip
from root import Root
import random
import math


class BaseSMA(Root):
    """
        Modified version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
        Notes:
            + Selected 2 unique and random solution to create new solution (not to create variable) --> remove third loop in original version
            + Check bound and update fitness after each individual move instead of after the whole population move in the original version
    """

    ID_WEI = 2

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, verbose=True, epoch=750, pop_size=100, z=0.03):
        Root.__init__(self, obj_func, lb, ub, problem_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size
        self.z = z

    def create_solution(self, minmax=1):
        pos = uniform(self.lb, self.ub)
        fit = self.get_fitness_position(pos,minmax)
        weight = zeros(self.problem_size)
        return [pos, fit, weight]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)      # Eq.(2.6)

        for epoch in range(self.epoch):

            s = pop[0][self.ID_FIT] - pop[-1][self.ID_FIT] + self.EPSILON  # plus eps to avoid denominator zero

            # calculate the fitness weight of each slime mold
            for i in range(0, self.pop_size):
                # Eq.(2.5)
                if i <= int(self.pop_size / 2):
                    pop[i][self.ID_WEI] = 1 + uniform(0, 1, self.problem_size) * log10((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)
                else:
                    pop[i][self.ID_WEI] = 1 - uniform(0, 1, self.problem_size) * log10((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)

            a = arctanh(-((epoch + 1) / self.epoch) + 1)                        # Eq.(2.4)
            b = 1 - (epoch + 1) / self.epoch

            # Update the Position of search agents
            for i in range(0, self.pop_size):
                if uniform() < self.z:  # Eq.(2.7)
                    pos_new = uniform(self.lb, self.ub)
                else:
                    p = tanh(abs(pop[i][self.ID_FIT] - g_best[self.ID_FIT]))    # Eq.(2.2)
                    vb = uniform(-a, a, self.problem_size)                      # Eq.(2.3)
                    vc = uniform(-b, b, self.problem_size)

                    # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                    id_a, id_b = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)

                    pos_1 = g_best[self.ID_POS] + vb * (pop[i][self.ID_WEI] * pop[id_a][self.ID_POS] - pop[id_b][self.ID_POS])
                    pos_2 = vc * pop[i][self.ID_POS]
                    pos_new = where(uniform(0, 1, self.problem_size) < p, pos_1, pos_2)

                # Check bound and re-calculate fitness after each individual move
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                pop[i][self.ID_POS] = pos_new
                pop[i][self.ID_FIT] = fit_new

            # Sorted population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, 1/g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalSMA(Root):
    """
        The original version of: Slime Mould Algorithm (SMA)
            (Slime Mould Algorithm: A New Method for Stochastic Optimization)
        Link:
            https://doi.org/10.1016/j.future.2020.03.055
    """

    def HHO(self, pop_i,all_pop,epoch,g_best):
        E1=2*(1-(epoch/self.epoch))
        E0=2*random.random()-1
        pop_i=numpy.array(pop_i)
        all_pop = numpy.array(all_pop)
        g_best = numpy.array(g_best)
        Escaping_Energy=E1*(E0)
        
        
        if abs(Escaping_Energy)>=1:
            #Harris' hawks perch randomly based on 2 strategy:
            q = random.random()
            rand_Hawk_index = math.floor(self.pop_size*random.random())
            print(all_pop[rand_Hawk_index])
            X_rand = all_pop[rand_Hawk_index][self.ID_POS]
            
            if q<0.5:
                # perch based on other family members
                pop_i[self.ID_POS]=X_rand-random.random()*abs(X_rand-2*random.random()*pop_i[self.ID_POS])

            elif q>=0.5:
                #perch on a random tall tree (random site inside group's home range)
                
                pop_i[self.ID_POS]=(g_best[self.ID_POS] - all_pop[:,self.ID_POS].mean(0))-random.random()*((self.ub-self.lb)*random.random()+self.lb)

        # -------- Exploitation phase -------------------
        elif abs(Escaping_Energy)<1:
            #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

            #phase 1: ----- surprise pounce (seven kills) ----------
            #surprise pounce (seven kills): multiple, short rapid dives by different hawks

            r=random.random() # probablity of each event
            
            if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                pop_i[self.ID_POS]=(g_best[self.ID_POS])-Escaping_Energy*abs(g_best[self.ID_POS]-pop_i[self.ID_POS])

            if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                pop_i[self.ID_POS]=(g_best[self.ID_POS]-pop_i[self.ID_POS])-Escaping_Energy*abs(Jump_strength*g_best[self.ID_POS]-pop_i[self.ID_POS])
            
            #phase 2: --------performing team rapid dives (leapfrog movements)----------
            if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                #rabbit try to escape by many zigzag deceptive motions
                Jump_strength=2*(1-random.random())
                X1=g_best[self.ID_POS]-Escaping_Energy*abs(Jump_strength*g_best[self.ID_POS]-pop_i[self.ID_POS])
                X1 = numpy.clip(X1, self.lb, self.ub)

                if self.compare_Best_bool(self.get_fitness_position(X1),g_best[self.ID_COMPARE]): # improved move?
                    pop_i[self.ID_POS] = X1.copy()
                else: # hawks perform levy-based short rapid dives around the rabbit
                    X2=g_best[self.ID_POS]-Escaping_Energy*abs(Jump_strength*g_best[self.ID_POS]-pop_i)+numpy.multiply(numpy.random.randn(self.problem_size),Levy(self.problem_size))
                    X2 = numpy.clip(X2, self.lb, self.ub)
                    if self.compare_Best_bool(self.get_fitness_position(X1),g_best[self.ID_COMPARE]):
                        pop_i[self.ID_POS] = X2.copy()
            if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                Jump_strength=2*(1-random.random())
                X1=g_best[self.ID_POS]-Escaping_Energy*abs(Jump_strength*g_best[self.ID_POS]-all_pop[:,self.ID_POS].mean(0))
                X1 = numpy.clip(X1, self.lb, self.ub)
                
                if self.compare_Best_bool(self.get_fitness_position(X1),g_best[self.ID_COMPARE]): # improved move?
                    pop_i[self.ID_POS] = X1.copy()
                else: # Perform levy-based short rapid dives around the rabbit
                    X2=g_best[self.ID_POS]-Escaping_Energy*abs(Jump_strength*g_best[self.ID_POS]-all_pop[:,self.ID_POS].mean(0))+numpy.multiply(numpy.random.randn(self.problem_size),self.Levy(self.problem_size))
                    X2 = numpy.clip(X2, self.lb, self.ub)
                    if self.compare_Best_bool(self.get_fitness_position(X1),g_best[self.ID_COMPARE]):
                        pop_i[self.ID_POS] = X2.copy()
        return pop_i
    ID_WEI = 2

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, verbose=True, epoch=750, pop_size=100, z=0.03):
        Root.__init__(self, obj_func, lb, ub, problem_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size
        self.z = z

    def create_solution(self, minmax=0):
        pos = uniform(self.lb, self.ub)
        fit = self.get_fitness_position(pos)
        weight = zeros(self.problem_size)
        return [pos, fit[2], weight,fit]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)      # Eq.(2.6)

        for epoch in range(self.epoch):
            s = pop[0][self.ID_FIT] - pop[-1][self.ID_FIT] + self.EPSILON       # plus eps to avoid denominator zero

            # calculate the fitness weight of each slime mold
            for i in range(0, self.pop_size):
                # Eq.(2.5)
                if i <= int(self.pop_size / 2):
                    pop[i][self.ID_WEI] = 1 + uniform(0, 1, self.problem_size) * log10((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)
                else:
                    pop[i][self.ID_WEI] = 1 - uniform(0, 1, self.problem_size) * log10((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)

            a = arctanh(-((epoch + 1) / self.epoch) + 1)                        # Eq.(2.4)
            b = 1 - (epoch + 1) / self.epoch

            # Update the Position of search agents
            for i in range(0, self.pop_size):
                if uniform() < self.z:                                          # Eq.(2.7)
                    pop[i][self.ID_POS] = uniform(self.lb, self.ub)
                else:
                    p = tanh(abs(pop[i][self.ID_FIT] - g_best[self.ID_FIT]))    # Eq.(2.2)
                    vb = uniform(-a, a, self.problem_size)                      # Eq.(2.3)
                    vc = uniform(-b, b, self.problem_size)
                    for j in range(0, self.problem_size):
                        # two positions randomly selected from population
                        id_a, id_b = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                        if uniform() < p:  # Eq.(2.1) addsion HHO
                            pop[i][self.ID_POS]=self.HHO(pop[i],pop,epoch,g_best)
                            break
                            # pop[i][self.ID_POS][j] = g_best[self.ID_POS][j] + vb[j] * (
                            #             pop[i][self.ID_WEI][j] * pop[id_a][self.ID_POS][j] - pop[id_b][self.ID_POS][j])
                        else:
                            pop[i][self.ID_POS][j] = vc[j] * pop[i][self.ID_POS][j]

            # Check bound and re-calculate fitness after the whole population move
            for i in range(0, self.pop_size):
                pos_new = self.amend_position(pop[i][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                pop[i][self.ID_POS] = pos_new
                pop[i][self.ID_FIT] = fit_new

            # Sorted population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
