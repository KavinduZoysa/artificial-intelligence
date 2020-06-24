import numpy as np
import matplotlib.pyplot as plt

high = 0.5
low = -0.5
loc = 0.0
sd = 0.08
x = 0.2
length = 100

z = np.random.normal(loc, sd, 100)
theta_minus = np.zeros(length)
theta_hat = np.zeros(length)

P = np.zeros(length)
P_minus = np.zeros(length)

K = np.zeros(length)

Q = 1e-5
R = 0.1**2

for k in range(1, length):
    theta_minus[k] = theta_hat[k-1]
    P_minus[k] = P[k-1] + Q
    K[k] = P_minus[k]/(P_minus[k] + R)
    theta_hat[k] = theta_minus[k] + K[k]*(z[k] - theta_minus[k])
    P[k] = P_minus[k] - K[k]*P_minus[k]

plt.figure()
plt.plot(z,'k+',label='noisy measurements')
plt.plot(theta_hat,'b-',label='a posteri estimate')
plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')
plt.show()
