import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import stats
from funcs import plot_fit4, round_sig_fig_uncertainty, fitting, quad, sqr, cubic
import scipy.optimize as opt

# import the data
data = ascii.read("part4_data.txt")

# gravitational acc
g = 9.80665 # m s^-2

# mass used
m = 24.8e-6 # kg

# C
C_data = ascii.read("c_values.txt")
C = C_data['C_redone'][0] # m^-1
C_err = C_data['C_redone_err'][0] # m^-1

# separate into different variables
F_up = data['m_up'] * g * 1e-3
I_up = data['I_up']
B_up = data['B_up']
B_up_err = data['B_up_err']

F_down = data['m_down'] * g * 1e-3
I_down = data['I_down']
B_down = data['B_down']
B_down_err = data['B_down_err']

F_down_neg = data['m_down_neg'] * g * 1e-3
I_down_neg = data['I_down_neg']
B_down_neg = data['B_down_neg']
B_down_neg_err = data['B_down_neg_err']

F_up_neg = data['m_up_neg'] * g * 1e-3
I_up_neg = data['I_up_neg']
B_up_neg = data['B_up_neg']
B_up_neg_err = data['B_up_neg_err']

I_up_err = np.ones_like(I_up)*0.01
I_up_err[I_up==0] = 0

I_down_err = np.ones_like(I_down)*0.01
I_down_err[I_down==0] = 0

I_down_neg_err = np.ones_like(I_down_neg)*0.01
I_down_neg_err[I_down_neg==0] = 0

I_up_neg_err = np.ones_like(I_up_neg)*0.01
I_up_neg_err[I_up_neg==0] = 0

Ferr = np.ones_like(F_up)*3e-6 * g

# fit the data to a linear graph
up_fit = stats.linregress(B_up, F_up)
down_fit = stats.linregress(B_down, F_down)
down_neg_fit = stats.linregress(B_down_neg, F_down_neg)
up_neg_fit = stats.linregress(B_up_neg, F_up_neg)

x = np.linspace(min(min(B_up), min(B_down)), max(max(B_up), max(B_down)), 2)
x_neg = np.linspace(min(min(B_up_neg), min(B_down_neg)), max(max(B_up_neg), max(B_down_neg)), 2)

# plot the data and fits
plt.errorbar(B_up, F_up, yerr=Ferr, xerr=B_up_err, fmt='none', color='blue')
plt.plot(x, x*up_fit.slope + up_fit.intercept, color='blue', label='Increasing B, I > 0 fit')
plt.scatter(B_up, F_up, s=5, color='blue', label='Increasing B, I > 0')
plot_fit4(up_fit)

plt.errorbar(B_down, F_down, yerr=Ferr, xerr=B_down_err, fmt='none', color='orange')
plt.plot(x, x*down_fit.slope + down_fit.intercept, color='orange', label='Decreasing B, I > 0 fit')
plt.scatter(B_down, F_down, s=5, color='orange', label='Decreasing B, I > 0')
plot_fit4(down_fit)

plt.errorbar(B_down_neg, F_down_neg, yerr=Ferr, xerr=B_down_neg_err, fmt='none', color='green')
plt.plot(x_neg, x_neg*down_neg_fit.slope + down_neg_fit.intercept, color='green', label='Decreasing B, I < 0 fit')
plt.scatter(B_down_neg, F_down_neg, s=5, color='green' , label='Decreasing B, I < 0')
plot_fit4(down_neg_fit)

plt.errorbar(B_up_neg, F_up_neg, yerr=Ferr, xerr=B_up_neg_err, fmt='none', color='red')
plt.plot(x_neg, x_neg*up_neg_fit.slope + up_neg_fit.intercept, color='red', label='Increasing B, I < 0 fit')
plt.scatter(B_up_neg, F_up_neg, s=5, color='red', label='Increasing B, I < 0')
plot_fit4(up_neg_fit)

plt.title("Plot of the measured excess vertical force due to the sample of Hematite ($F_Z$) against the magnetic field stength ($B$)")
plt.xlabel('$B$ [$T$]')
plt.ylabel('$F_Z$ [N]')
plt.legend(ncols=2)
plt.grid()
plt.show()

# Each slope is the value of C m sigma so calculate sigma easily
slopes = np.array([up_fit.slope, down_fit.slope, down_neg_fit.slope, up_neg_fit.slope])

sigma = slopes/(m*C)

# error is sqrt( (1/mC)^2 * Dslope^2 + (-slope/m^2C)^2 * Dm^2 + (-slope/mC^2)^2 * DC^2 )
slope_err = np.array([up_fit.stderr, down_fit.stderr, down_neg_fit.stderr, up_neg_fit.stderr])

m_err = 5e-7 # kg

sigma_errs = np.sqrt( (1/(m*C))**2 * slope_err**2 + (-slopes/(m*C))**2 * m_err**2 + (-slopes/(m*C))**2 * C_err**2   )

sigma, sigma_errs = round_sig_fig_uncertainty(sigma, sigma_errs)


print(f"The value of \u03C3 calculated while increasing B, I > 0: {sigma[0]} \u00B1 {sigma_errs[0]} J T\u207B\u00B9 kg\u207B\u00B9")
print(f"The value of \u03C3 calculated while decreasing B, I > 0: {sigma[1]} \u00B1 {sigma_errs[1]} J T\u207B\u00B9 kg\u207B\u00B9")
print(f"The value of \u03C3 calculated while decreasing B, I < 0: {sigma[2]} \u00B1 {sigma_errs[2]} J T\u207B\u00B9 kg\u207B\u00B9")
print(f"The value of \u03C3 calculated while increasing B, I < 0: {sigma[3]} \u00B1 {sigma_errs[3]} J T\u207B\u00B9 kg\u207B\u00B9")

# Try non-linear fits
# Quadratic
p0 = np.array([0.01, 0.01])

up_fit = fitting(p0, B_up, F_up, quad, B_up_err, Ferr, 'blue', 'Increasing B, I > 0')
down_fit = fitting(p0, B_down, F_down, quad, B_down_err, Ferr, 'orange', 'Decreasing B, I > 0')
down_neg_fit = fitting(p0, B_down_neg, F_down_neg, quad, B_down_neg_err, Ferr, 'green', 'Decreasing B, I < 0')
up_neg_fit = fitting(p0, B_up_neg, F_up_neg, quad, B_up_neg_err, Ferr, 'red', 'Increasing B, I < 0')

plt.title("Plot of the measured excess vertical force due to the sample of Hematite ($F_Z$) against the magnetic field stength ($B$) with a quadratic fit")
plt.xlabel('$B$ [$T$]')
plt.ylabel('$F_Z$ [N]')
plt.legend()
plt.grid()
plt.show()

quad_out = open('quad_fit_file.txt','w', encoding='utf-8')
quad_out.write('F = aB^2 + bB\n')
quad_out.write('a\tb\tchi2\n')

fits = [up_fit, down_fit, down_neg_fit, up_neg_fit]

for i in range(len(fits)):
    for j in range(len(fits[i][0])):
        quad_out.write(str(fits[i][0][j])  + '\t')
    quad_out.write(str(fits[i][1]) + '\n')


# find the new values of B
# need to import the coefficients from part 1
coeff_data = ascii.read("coefficients_file.txt")

B_up_fit = np.zeros_like(B_up)
for j in range(len(coeff_data[0])):
    B_up_fit = B_up_fit + coeff_data[0][j] * I_up**j

B_down_fit = np.zeros_like(B_down)
for j in range(len(coeff_data[1])):
    B_down_fit = B_down_fit + coeff_data[1][j] * I_down**j

B_down_neg_fit = np.zeros_like(B_down_neg)
for j in range(len(coeff_data[2])):
    B_down_neg_fit = B_down_neg_fit + coeff_data[2][j] * I_down_neg**j

B_up_neg_fit = np.zeros_like(B_up_neg)
for j in range(len(coeff_data[3])):
    B_up_neg_fit = B_up_neg_fit + coeff_data[3][j] * I_up_neg**j

plt.scatter(I_up, B_up_fit, label='Increasing B, I > 0', color='blue')
plt.scatter(I_down, B_down_fit, label='Decreasing B, I > 0', color='orange')
plt.scatter(I_down_neg, B_down_neg_fit, label='Decreasing B, I < 0', color='green')
plt.scatter(I_up_neg, B_up_neg_fit, label='Decreasing B, I < 0', color='red')


plt.xlabel('I [A]')
plt.ylabel('B [T]')
plt.title('The fitted curve of the magnetic field strength (B) as a function of the current (I)')
plt.grid()
plt.legend()
plt.show()

# now find sigma(B)
sigma_up = (B_up_fit*up_fit[0][0] + up_fit[0][1])/(m*C)
sigma_down = (B_down_fit*down_fit[0][0] + down_fit[0][1])/(m*C)
sigma_down_neg = (B_down_neg_fit*down_neg_fit[0][0] + down_neg_fit[0][1])/(m*C)
sigma_up_neg = (B_up_neg_fit*up_neg_fit[0][0] + up_neg_fit[0][1])/(m*C)



plt.scatter(B_up_fit, sigma_up, label='Increasing B, I > 0', color='blue')
plt.scatter(B_down_fit, sigma_down, label='Decreasing B, I > 0', color='orange')
plt.scatter(B_down_neg_fit, sigma_down_neg, label='Decreasing B, I < 0', color='green')
plt.scatter(B_up_neg_fit, sigma_up_neg, label='Decreasing B, I < 0', color='red')


plt.xlabel('B [T]')
plt.ylabel('$\sigma$ [J T$^{-1}$ kg$^{-1}$]')
plt.title('The fitted curve of $\sigma$ (B) for the ferromagnetic hematite')
plt.grid()
plt.legend()
plt.show()

# try cubic
p0 = np.array([1, 1, 1])

up_fit = fitting(p0, B_up, F_up, cubic, B_up_err, Ferr, 'blue', 'Increasing B, I > 0')
down_fit = fitting(p0, B_down, F_down, cubic, B_down_err, Ferr, 'orange', 'Decreasing B, I > 0')
down_neg_fit = fitting(p0, B_down_neg, F_down_neg, cubic, B_down_neg_err, Ferr, 'green', 'Decreasing B, I < 0')
up_neg_fit = fitting(p0, B_up_neg, F_up_neg, cubic, B_up_neg_err, Ferr, 'red', 'Increasing B, I < 0')

plt.title("Plot of the measured excess vertical force due to the sample of Hematite ($F_Z$) against the magnetic field stength ($B$) with a cubic fit")
plt.xlabel('$B$ [$T$]')
plt.ylabel('$F_Z$ [N]')
plt.legend()
plt.grid()
plt.show()

cubic_out = open('cubic_fit_file.txt','w', encoding='utf-8')
cubic_out.write('F = aB^3 + bB^2 + cB\n')
cubic_out.write('a\tb\tc\tchi2\n')

fits = [up_fit, down_fit, down_neg_fit, up_neg_fit]

for i in range(len(fits)):
    for j in range(len(fits[i][0])):
        cubic_out.write(str(fits[i][0][j])  + '\t')
    cubic_out.write(str(fits[i][1]) + '\n')


# now find sigma(B)
sigma_up = (B_up_fit**2*up_fit[0][0] + B_up_fit*up_fit[0][1] + up_fit[0][2])/(m*C)
sigma_down = (B_down_fit**2*down_fit[0][0] + B_down_fit*down_fit[0][1] + down_fit[0][2])/(m*C)
sigma_down_neg = (B_down_neg_fit**2*down_neg_fit[0][0] + B_down_neg_fit*down_neg_fit[0][1] + down_neg_fit[0][2])/(m*C)
sigma_up_neg = (B_up_neg_fit**2*up_neg_fit[0][0] + B_up_neg_fit*up_neg_fit[0][1] + up_neg_fit[0][2])/(m*C)



plt.scatter(B_up_fit, sigma_up, label='Increasing B, I > 0', color='blue')
plt.scatter(B_down_fit, sigma_down, label='Decreasing B, I > 0', color='orange')
plt.scatter(B_down_neg_fit, sigma_down_neg, label='Decreasing B, I < 0', color='green')
plt.scatter(B_up_neg_fit, sigma_up_neg, label='Decreasing B, I < 0', color='red')


plt.xlabel('B [T]')
plt.ylabel('$\sigma$ [J T$^{-1}$ kg$^{-1}$]')
plt.title('The fitted curve of $\sigma$ (B) for the ferromagnetic hematite')
plt.grid()
plt.legend()
plt.show()



'''
# Try just squared
p0 = 0.01

up_fit = fitting(p0, B_up, F_up, sqr, B_up_err, Ferr, 'blue', 'Increasing B, I > 0')
down_fit = fitting(p0, B_down, F_down, sqr, B_down_err, Ferr, 'orange', 'Decreasing B, I > 0')
down_neg_fit = fitting(p0, B_down_neg, F_down_neg, sqr, B_down_neg_err, Ferr, 'green', 'Decreasing B, I < 0')
up_neg_fit = fitting(p0, B_up_neg, F_up_neg, sqr, B_up_neg_err, Ferr, 'red', 'Increasing B, I < 0')

plt.title("Plot of the measured excess vertical force due to the sample of Hematite ($F_Z$) against the magnetic field stength ($B$)")
plt.xlabel('$B$ [$T$]')
plt.ylabel('$F_Z$ [N]')
plt.legend()
plt.grid()
plt.show()

sqr_out = open('sqr_fit_file.txt','w', encoding='utf-8')
sqr_out.write('F = aB^2\n')
sqr_out.write('a\tchi2\n')

fits = [up_fit, down_fit, down_neg_fit, up_neg_fit]

for i in range(len(fits)):
    sqr_out.write(str(fits[i][0])  + '\t')
    sqr_out.write(str(fits[i][1]) + '\n')

# now find sigma(B)
sigma_up = (B_up_fit*up_fit[0])/(m*C)
sigma_down = (B_down_fit*down_fit[0])/(m*C)
sigma_down_neg = (B_down_neg_fit*down_neg_fit[0])/(m*C)
sigma_up_neg = (B_up_neg_fit*up_neg_fit[0])/(m*C)

plt.scatter(B_up_fit, sigma_up)
plt.scatter(B_down_fit, sigma_down)
plt.scatter(B_down_neg_fit, sigma_down_neg)
plt.scatter(B_up_neg_fit, sigma_up_neg)
plt.show()
'''