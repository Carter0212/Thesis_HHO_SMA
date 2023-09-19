import numpy as np
import random
import matplotlib.pyplot as plt
iteration = 10000
Escaping_Energy_list = []
E1_list = []
E0_list = []
Escaping_Energy_list2 = []
for t in range(iteration):
    E1_2 = np.log(np.power(t/iteration,2))
    E1=2*(1-(t/iteration))
    E0=2*random.random()-1  # -1<E0<1
    Escaping_Energy=E1*(E0)
    Escaping_Energy2=E1_2*E0
    E1_list.append(E1)
    E0_list.append(E0)
    Escaping_Energy_list.append(Escaping_Energy)
    Escaping_Energy_list2.append(Escaping_Energy2)

fig_plot, ax_plot = plt.subplots()
ax_plot.plot(Escaping_Energy_list2,label=f'Escaping_Energy',color='black')
ax_plot.plot(Escaping_Energy_list,label=f'Escaping_Energy',color='red')

# ax_plot.plot(E0_list,label=f'E0',color='black')
# ax_plot.plot(E1_list,label=f'E1',color='blue')
ax_plot.legend()
fig_plot.savefig('a.jpg')