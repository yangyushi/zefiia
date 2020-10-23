import zefiia as za
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 14


N = 100
x = np.linspace(0, 10, N)
par = [1e-5, 2e-4, -3e-3, -2e-2, 1e-1, 1]
y = np.polyval(par, x)
y_noise = y + np.random.uniform(-0.1, 0.1, N)

outliers = np.array((
    np.random.uniform(0, 10, N // 2),
    np.random.uniform(-1, 3, N // 2)
))
features = np.array((x, y_noise))
features = np.concatenate((features, outliers), axis=1)
features_ransac = za.ransac(features.T, degree=5, sigma=0.1, N=25, T=N//2)
par_est = np.polyfit(*features_ransac.T, deg=5)

plt.plot(x, y, color='k', lw=3)
plt.plot(x, y, color='pink', lw=2, label='true model')

plt.plot(x, np.polyval(par_est, x), color='k', lw=3, ls='-')
plt.plot(x, np.polyval(par_est, x), color='lightblue', lw=2, ls='-', label='ransac model')

plt.scatter(x, y_noise, color='teal', fc='none', s=120, label='noisy data')
plt.scatter(*outliers, color='silver', marker='x', label='outliers')
plt.scatter(*features_ransac.T, color='tomato', marker='o', label='ransac')
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('ransac.pdf')