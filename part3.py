import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import stats
from funcs import plot_fit, round_sig_fig_uncertainty

# import the data
data = ascii.read("part3_data.txt")

# gravitational acc
g = 9.80665 # m s^-2

# C
C_data = ascii.read("c_values.txt")
C_up = C_data['C_up'][0] # m^-1
C_up_err = C_data['C_up_err'][0] # m^-1
C_down = C_data['C_down'][0] # m^-1
C_down_err = C_data['C_down_err'][0] # m^-1

C_arr = np.array([C_up, C_down])
C_err_arr = np.array([C_up_err, C_down_err])

# mass used
m = 14.8e-6 # kg

# separate into different variables
F_up = data['m_up'] * g * 1e-3
B_up = data['B_up']
B_up_err = data['B_up_err']

F_down = data['m_down'] * g * 1e-3
B_down = data['B_down']
B_down_err = data['B_down_err']

Ferr = np.ones_like(F_up)*1e-7 * g

# fit the data to a linear graph
up_fit = stats.linregress(B_up[5:], F_up[5:])
down_fit = stats.linregress(B_down[:-3], F_down[:-3])

x = np.linspace(min(min(B_up), min(B_down)), max(max(B_up), max(B_down)), 2)

# plot the data and fits
plt.errorbar(B_up, F_up, yerr=Ferr, xerr=B_up_err, fmt='none', color='blue')
plt.plot(x, x*up_fit.slope + up_fit.intercept, color='blue', label='Increasing B, I > 0 fit')
plt.scatter(B_up, F_up, s=5, color='blue', label='Increasing B, I > 0')
plot_fit(up_fit)

plt.errorbar(B_down, F_down, yerr=Ferr, xerr=B_down_err, fmt='none', color='green')
plt.plot(x, x*down_fit.slope + down_fit.intercept, color='green', label='Decreasing B, I > 0 fit')
plt.scatter(B_down, F_down, s=5, color='green', label='Decreasing B, I > 0')
plot_fit(down_fit)

plt.title("Plot of the measured excess vertical force due to the sample of $Gd_3Ga_5O_{12}$ ($F_Z$) against the magnetic field stength squared ($B^2$)")
plt.xlabel('$B^2$ [$T^2$]')
plt.ylabel('$F_Z$ [N]')
plt.legend()
plt.grid()
plt.show()


# Each slope is the value of C m chi so calculate chi easily
chi_up = up_fit.slope/(m*C_up)
chi_down = down_fit.slope/(m*C_down)

chi_arr = np.array([chi_up, chi_down])

# error is sqrt( (1/mC)^2 * Dslope^2 + (-slope/m^2C)^2 * Dm^2 + (-slope/mC^2)^2 * DC^2 )
slope = [up_fit.slope, down_fit.slope]
slope_err = [up_fit.stderr, down_fit.stderr]

m_err = 5e-7 # kg

chi_err_arr = np.array([])

for i in range(len(chi_arr)):
    chi_err = np.sqrt( (1/(m*C_arr[i]))**2 * slope_err[i]**2 + (-slope[i]/(m*C_arr[i]))**2 * m_err**2 + (-slope[i]/(m*C_arr[i]))**2 * C_err_arr[i]**2   )
    chi_err_arr = np.append(chi_err_arr, chi_err)

# round the values and print them
    
print(chi_arr, chi_err_arr)
chi_arr, chi_err_arr = round_sig_fig_uncertainty(chi_arr, chi_err_arr)

print(f"The value of \u03C7 calculated while increasing B: {chi_arr[0]} \u00B1 {chi_err_arr[0]} J T\u207B\u00B2 kg\u207B\u00B9")
print(f"The value of \u03C7 calculated while decreasing B: {chi_arr[1]} \u00B1 {chi_err_arr[1]} J T\u207B\u00B2 kg\u207B\u00B9")
