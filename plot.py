import numpy as np
import matplotlib.pyplot as plt
fig_EE,ax_EE = plt.subplots()
d=np.arange(0,201)
# d_45=d/45
# a=np.exp(d_45)
# g=np.where(24/d < 1.0, 24/d, 1)
print(np.where(24/d < 1.0, 24/d, 1))
# P=np.where(18/d < 1.0, 18/d, 1)+np.exp(-d/36)*(1-18/d)
P=np.where(24/d < 1.0, 24/d, 1)*(1-np.exp(-d/45))+np.exp(-d/45)
print(P)
ax_EE.plot(P)
plt.show()
