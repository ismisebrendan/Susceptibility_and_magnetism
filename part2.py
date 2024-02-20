import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import stats
from funcs import plot_fit, round_sig_fig_uncertainty

# import the data
data = ascii.read("part2_data.txt")

# gravitational acc
g = 9.80665 # m s^-2

# chi
chi = 0.330 # J T^-2 kg^-1 (at 20 C)

# mass used
m = 63.5e-6 # kg

# separate into different variables
F_up = data['m_up'] * g * 1e-3
B_up = data['B_up']
B_up_err = np.abs(data['B_up_err'])

F_down = data['m_down'] * g * 1e-3
B_down = data['B_down']
B_down_err = np.abs(data['B_down_err'])

F_down_neg = data['m_down_neg'] * g * 1e-3
B_down_neg = data['B_down_neg']
B_down_neg_err = np.abs(data['B_down_neg_err'])

F_up_neg = data['m_up_neg'] * g * 1e-3
B_up_neg = data['B_up_neg']
B_up_neg_err = np.abs(data['B_up_neg_err'])

Ferr = np.ones_like(F_up)*1e-7 * g

# fit the data to a linear graph
up_fit = stats.linregress(B_up[19:], F_up[19:])
down_fit = stats.linregress(B_down, F_down)
down_neg_fit = stats.linregress(B_down_neg, F_down_neg)
up_neg_fit = stats.linregress(B_up_neg, F_up_neg)

x = np.linspace(min(min(B_up), min(B_down), min(B_down_neg), min(B_up_neg)), max(max(B_up), max(B_down), max(B_down_neg), max(B_up_neg)), 2)

# plot the data and fits
plt.plot(x, x*up_fit.slope + up_fit.intercept, color='blue', label='Increasing B, I > 0 fit')
plt.errorbar(B_up, F_up, yerr=Ferr, xerr=B_up_err, fmt='none', color='blue', label='Increasing B, I > 0')
plot_fit(up_fit)

plt.plot(x, x*down_fit.slope + down_fit.intercept, color='orange', label='Decreasing B, I > 0 fit')
plt.errorbar(B_down, F_down, yerr=Ferr, xerr=B_down_err, fmt='none', color='orange', label='Decreasing B, I > 0')
plot_fit(down_fit)

plt.plot(x, x*down_neg_fit.slope + down_neg_fit.intercept, color='green', label='Decreasing B, I < 0 fit')
plt.errorbar(B_down_neg, F_down_neg, yerr=Ferr, xerr=B_down_neg_err, fmt='none', color='green', label='Decreasing B, I < 0')
plot_fit(down_neg_fit)

plt.plot(x, x*up_neg_fit.slope + up_neg_fit.intercept, color='red', label='Increasing B, I < 0 fit')
plt.errorbar(B_up_neg, F_up_neg, yerr=Ferr, xerr=B_up_neg_err, fmt='none', color='red', label='Increasing B, I < 0')
plot_fit(up_neg_fit)

plt.title("Plot of the measured excess vertical force due to the sample of Mohr's salt ($F_Z$) against the magnetic field stength squared ($B^2$)")
plt.xlabel('$B^2$ [$T^2$]')
plt.ylabel('$F_Z$ [N]')

handles, labels = plt.gca().get_legend_handles_labels()

order = [20, 0, 1, 2, 3, 4, 21, 5, 6, 7, 8, 9, 22, 10, 11, 12, 13, 14, 23, 15, 16, 17, 18, 19]

plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncols=2) 

plt.grid()
plt.show()

# Each slope is the value of C m chi, so calculate C easily
C_up = up_fit.slope/(m*chi)
C_down = down_fit.slope/(m*chi)
C_down_neg = down_neg_fit.slope/(m*chi)
C_up_neg = up_neg_fit.slope/(m*chi)

C_arr = np.array([C_up, C_down, C_down_neg, C_up_neg])

# error is sqrt( (1/mchi)^2 * Dslope^2 + (-slope/m^2chi)^2 * Dm^2 + (-slope/mchi^2)^2 * Dchi^2 )
slope = [up_fit.slope, down_fit.slope, down_neg_fit.slope, up_neg_fit.slope]
slope_err = [up_fit.stderr, down_fit.stderr, down_neg_fit.stderr, up_neg_fit.stderr]

m_err = 5e-7 # kg
chi_err = 0.001 # J T^-2 kg^-1 (at 20 C)

C_err_arr = np.array([])

for i in range(len(C_arr)):
    C_err = np.abs(np.sqrt( (1/(m*chi))**2 * slope_err[i]**2 + (-slope[i]/(m*chi))**2 * m_err**2 + (-slope[i]/(m*chi))**2 * chi_err**2   ))
    C_err_arr = np.append(C_err_arr, C_err)

# round the values and export them
C_arr, C_err_arr = round_sig_fig_uncertainty(C_arr, C_err_arr)

C_file = open('c_values.txt','w', encoding='utf-8')

C_file.write('C_up\tC_up_err\tC_down\tC_down_err\tC_down_neg\tC_down_neg_arr\tC_up_neg\tC_up_neg_err\tC_mean\tC_mean_err\tC_redone\tC_redone_err\n')

for C in range(len(C_arr)):
    C_file.write(str(C_arr[C]) +'\t' + str(C_err_arr[C]) +'\t')

C_file.write(str(np.mean(C_arr)) +  '\t' + str(np.std(C_arr)))

## Now the redone part
data = ascii.read("part2_redone_data.txt")

m = 53.9e-6 # kg

# separate into different variables
F = data['m'] * g * 10**-3
B = data['B']
Berr = np.abs(data['Berr'])
Ferr = np.ones_like(F)*1e-7 * g

# fit the data to a linear graph
fit = stats.linregress(B, F)

x = np.linspace(B[0], B[-1], 2)

# plot the data and fits
plt.errorbar(B, F, yerr=Ferr, xerr=Berr, fmt='none', color='blue', label='Increasing B, I > 0')
plt.plot(x, x*fit.slope + fit.intercept, color='blue', label='Increasing B, I > 0 fit')
plot_fit(fit)

plt.title("Plot of the measured excess vertical force due to the sample of Mohr's salt ($F_Z$) against the magnetic field stength squared ($B^2$)")
plt.xlabel('$B^2$ [$T^2$]')
plt.ylabel('$F_Z$ [N]') 

handles, labels = plt.gca().get_legend_handles_labels()

order = [5, 0, 1, 2, 3, 4]

plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.grid()
plt.show()

# The slope is the value of C m chi
C = fit.slope/(m*chi)
C_err = np.sqrt( (1/(m*chi))**2 * fit.stderr**2 + (-fit.slope/(m*chi))**2 * m_err**2 + (-fit.slope/(m*chi))**2 * chi_err**2   )


# round the values and export them
C, C_err = round_sig_fig_uncertainty(C, C_err)
C_file.write('\t' + str(C) + '\t' + str(C_err))