from numpy import where, clip, logical_and, ones, array, ceil,shape
from numpy.random import uniform
from copy import deepcopy
import functools
import numpy as np

class Root:
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0         # min problem
    ID_MAX_PROB = -1        # max problem

    ID_POS = 0              # Position
    ID_FIT = 1              # Fitness
    ID_COMPARE = 3
    EPSILON = 10E-10 

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, verbose=True):
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
        self.obj_func = obj_func
        if (lb is None) or (ub is None):
            if problem_size is None:
                print("Problem size must be an int number")
                exit(0)
            elif problem_size <= 0:
                print("Problem size must > 0")
                exit(0)
            else:
                self.problem_size = int(ceil(problem_size))
                self.lb = -1 * ones(problem_size)
                self.ub = 1 * ones(problem_size)
        else:
            if isinstance(lb, list) and isinstance(ub, list) and not (problem_size is None):
                if (len(lb) == len(ub)) and (problem_size > 0):
                    if len(lb) == 1:
                        self.problem_size = problem_size
                        self.lb = lb[0] * ones(problem_size)
                        self.ub = ub[0] * ones(problem_size)
                    else:
                        self.problem_size = len(lb)
                        self.lb = array(lb)
                        self.ub = array(ub)
                else:
                    print("Lower bound and Upper bound need to be same length. Problem size must > 0")
                    exit(0)
            else:
                print("Lower bound and Upper bound need to be a list. Problem size is an int number")
                exit(0)
        self.verbose = verbose
        self.epoch, self.pop_size = None, None
        self.solution, self.loss_train = None, []

    def create_solution(self, minmax=1):
        """ Return the position position with 2 element: position of position and fitness of position
        Parameters
        ----------
        minmax
            0 - minimum problem, else - maximum problem
        """
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        return [position, fitness[2],fitness]

    def get_fitness_position(self, position=None, minmax=1):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :param minmax: 0- min problem, 1 - max problem
        :return:
        """
        # print(self.obj_func(position))
        # print(1.0 / (self.obj_func(position) + self.EPSILON))
        # return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + self.EPSILON)
        ans=self.obj_func(position)
        # ans=(ans[0],ans[1],ans[2],ans[3],ans[4],ans[5])
        return ans
    
    def compare_Best_min(self,news,olds):
        if news > olds:
            return 1
        return -1
    
    
    def compare_Best(self,news,olds):
        news=news[3]
        olds=olds[3]
        if news[0]==True and olds[0] == True and news[2]>olds[2]:
            return -1
        if news[0]==True and olds[0] == False:
            return -1
        if news[0]==False and olds[0] == False and news[3] < olds[3]:
            return -1
        return 1

    def compare_Best_bool(self,news,olds):
        if news[0]==True and olds[0] == True and news[2]>olds[2]:
            return True
        if news[0]==True and olds[0] == False:
            return True
        if news[0]==False and olds[0] == False and news[3] < olds[3]:
            return True
        return False
    
    def f2_compare_Best(self,news,olds):
        if news < olds:
            return -1
        return 1
    def f2_compare_Best_bool(self,news,olds):
        if news < olds:
            return True
        return False

    def get_sorted_pop_and_global_best_solution(self, pop=None, id_fit=None, id_best=None , id_comapre=None):
        """ Sort population and return the sorted population and the best position """
        # sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        print('=============================')
        
        # sorted_pop = sorted(pop, key=functools.cmp_to_key(self.compare_Best_min))
        # print('-----------------------------')
        # for i in sorted_pop:
        #     print(i[3])
        # print('=============================')
        cmp_func=functools.cmp_to_key(self.f2_compare_Best)
        np_cmp_func =np.vectorize(cmp_func)
        pop=np.array(pop)
        sort_data=np_cmp_func(pop[:,self.ID_FIT])
        ans_index=np.argsort(sort_data)
        sorted_pop=pop[ans_index]
        sorted_pop=sorted_pop[:self.problem_size]
        #############
        g_best = sorted_pop[id_best]
        sorted_pop=sorted_pop.tolist()
        return sorted_pop, g_best
        # for pop in sorted_pop:
        #     print(pop[3])
        # exit(1)
        return sorted_pop, deepcopy(sorted_pop[id_best])

    def amend_position(self, position=None):
        return clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return where(logical_and(self.lb <= position, position <= self.ub), position, uniform(self.lb, self.ub))

    def update_sorted_population_and_global_best_solution(self, pop=None, id_best=None, g_best=None):
        """ Sort the population and update the current best position. Return the sorted population and the new current best position """
        # sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        cmp_func=functools.cmp_to_key(self.f2_compare_Best)
        np_cmp_func =np.vectorize(cmp_func)
        pop=np.array(pop)
        sort_data=np_cmp_func(pop[:,self.ID_FIT])
        ans_index=np.argsort(sort_data)
        sorted_pop=pop[ans_index]
        sorted_pop=sorted_pop[:self.problem_size]
        #############
        current_best = sorted_pop[id_best]
        sorted_pop=sorted_pop.tolist()
        # g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)
        g_best = deepcopy(current_best) if self.f2_compare_Best_bool(current_best[self.ID_COMPARE],g_best[self.ID_COMPARE])  else deepcopy(g_best)
        return sorted_pop, g_best
