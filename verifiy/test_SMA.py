from SMA import BaseSMA, OriginalSMA
from numpy import sum, pi, exp, sqrt, cos


## You can create whatever function you want here
def func_sum(solution):
    return sum(solution ** 2)


def func_ackley(solution):
    a, b, c = 20, 0.2, 2 * pi
    d = len(solution)
    sum_1 = -a * exp(-b * sqrt(sum(solution ** 2) / d))
    sum_2 = exp(sum(cos(c * solution)) / d)
    return sum_1 - sum_2 + a + exp(1)

def f1(x):
    ans=1
    for i in x:
        ans*=abs(i)
    return sum(abs(num) for num in x) + ans
    # return sum(num**2 for num in x)

## You can create different bound for each dimension like this
# lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -100, -40, -50]
# ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 20, 200, 1000]
# problem_size = 18
## if you choose this way, the problem_size need to be same length as lb and ub

## Or bound is the same for all dimension like this
lb = [-10]
ub = [10]
problem_size = 30
## if you choose this way, the problem_size can be anything you want

    # dim_list=[30,100,500,1000]
    # SearchAgents_no=30
    # lb=10
    # ub=10
    # Max_iter=500
## Setting parameters
obj_func = f1
verbose = True
epoch = 3000 ## iterators
pop_size = 30

md1 = BaseSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
# return : the global best solution, the fitness of global best solution and the loss of training process in each epoch/iteration
print(f'BaseSMA:{md1.solution[0]}')
print(f'BaseSMA:{md1.solution[1]}')
# print(md1.loss_train)

md2 = OriginalSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
best_pos2, best_fit2, list_loss2 = md2.train()
# return : the global best solution, the fitness of global best solution and the loss of training process in each epoch/iteration
print(f'OriginalSMA:{best_pos2}')
print(f'OriginalSMA:{best_fit2}')
# print(list_loss2)
