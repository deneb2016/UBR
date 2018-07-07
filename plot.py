import numpy as np
import matplotlib.pyplot as plt

plt.xscale('log')
plt.grid(True)
x = []
for i in [1, 10, 100, 1000]:
    for j in range(1, 10):
        x.append(i * j)
x.append(10000)

UBR = [0.2, 0.3, ]

plt.plot(x, [50, 70, 80])
plt.show()