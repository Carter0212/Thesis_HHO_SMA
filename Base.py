import abc
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import random
import math
from solution import solution
import numpy as np
import functools
from typing import Optional, Callable, Any

class Root(abc.ABC):
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0         # min problem
    ID_MAX_PROB = -1        # max problem

    ID_POS = 0              # Position
    ID_FIT = 1              # Fitness
    ID_COMPARE = 3
    EPSILON = 10E-10

    def __init__(self,function,dimension,iteration,problem_size,lb,ub):
        """
        Parameters
        ----------
        obj_func : function
        lb : list
        ub : list
        problem_size : int, optional
        batch_size: int, optional
        verbose : bool, optional
        """
        # assert (type(variable_type_mixed).__module__=='numpy'),\
        #     "\n variable_type must be numpy array"
        
        #############################################################
        # check all parameters
        assert(type(problem_size) is int),"Problem size must be an int number"
        assert(problem_size > 0),"Problem size must > 0"
        assert((ub is not None) and (lb is not None)),"You must seeting your low bound and upper bound."
        assert(dimension > 0),"dimension must > 0"
        assert(function),"pls setting your obj_fuction"
        assert(iteration > 0),"problem_size must > 0"
        #############################################################


        #############################################################
        # update all parameters to self
        self.dimension = dimension
        self.problem_size = problem_size ## SearchAgents_no
        self.iteration = iteration
        self.obj_func = function
        self.ub =ub
        self.lb =lb
        #############################################################
        
        self.init_params()



    
    def create_solution(self, minmax=0):
        # if not isinstance(self.lb, list):
        #     self.lb = [self.lb for _ in range(self.dimension)]
        #     self.ub = [self.ub for _ in range(self.dimension)]
        # self.lb = np.asarray(self.lb)
        # self.ub = np.asarray(self.ub)
        pos = np.asarray([int (x*(self.ub-self.lb)+self.lb) for x in np.random.uniform(0,1,self.dimension)])
        fit = self.get_fitness(pos)
        weight = np.zeros(self.problem_size)
        return (pos, fit)
    
    def get_fitness(self, position=None, minmax=1):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :param minmax: 0- min problem, 1 - max problem
        :return:
        """
        # print(self.obj_func(position))
        # print(1.0 / (self.obj_func(position) + self.EPSILON))
        # return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + self.EPSILON)
        
        return self.obj_func(position)
        # ans=(ans[0],ans[1],ans[2],ans[3],ans[4],ans[5])

    def get_sorted_pop_and_global_best_solution(self, pop=None, id_fit=None, id_best=None,sort_func=None):
        """ Sort population and return the sorted population and the best position """
        cmp_func=functools.cmp_to_key(sort_func)
        np_cmp_func =np.vectorize(cmp_func)
        sort_data=np_cmp_func(pop[:,1])
        ans_index=np.argsort(sort_data)
        sorted_pop=pop[ans_index]
        return sorted_pop, sorted_pop[id_best].copy()
    
    def update_sorted_population_and_global_best_solution(self, pop=None, id_best=None, g_best=None,sort_func=None,compare_func_bool=None,original_pop=None):
        """ Sort the population and update the current best position. Return the sorted population and the new current best position """
        combin_pop=np.append(pop,original_pop,axis=0)

        cmp_func=functools.cmp_to_key(sort_func)
        np_cmp_func =np.vectorize(cmp_func)
        sort_data=np_cmp_func(combin_pop[:,1])
        ans_index=np.argsort(sort_data)
        sorted_pop=combin_pop[ans_index]
        sorted_pop=sorted_pop[:self.problem_size]
        #############
        current_best = sorted_pop[id_best]
        if compare_func_bool(current_best[self.ID_FIT],g_best[self.ID_FIT]):
            g_best = current_best.copy()
            return sorted_pop, g_best
        return sorted_pop, g_best
    
    def amend_position(self, position=None):
        return np.clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return np.where(np.logical_and(self.lb <= position, position <= self.ub), position, np.random.uniform(self.lb, self.ub))
    
    # def compare_Best_bool(self,news,olds):
    #     if news[0]==True and olds[0] == True and news[2]>olds[2]:
    #         return True
    #     if news[0]==True and olds[0] == False:
    #         return True
    #     if news[0]==False and olds[0] == False and news[3] < olds[3]:
    #         return True
    #     return False

    @abc.abstractmethod
    def run(self):
        'Return algorithms result'
        return NotImplemented

    
    def mutil_run(self, run_times: int, aggrigator: Optional[Callable[[Any], None]] = None) -> None:
        for i in range(run_times):
            self.run()
            if aggrigator is not None:
                aggrigator(self, ...)
    
    def init_params(self):
        self.startTime=None 
        self.endTime=None
        self.executionTime=None ## algorithm execution time
        self.convertTime=None
        self.optimizer=None
        self.objfname=None
        self.best=None
        self.bestIndividual=None
        self.ALLData=None

        
    def show_time(self):
        print(self.executionTime)

    def get_best_individual(self):
        pass

    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()