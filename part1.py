import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import interpolate
from funcs import poly

# import the data
data = ascii.read("part1_data.txt")

# separate into different variables
I_up = data['I_up']
B_up = data['B_up']
B_up_err = data['B_up_err']

I_down = np.flip(data['I_down'])
B_down = np.flip(data['B_down'])
B_down_err = np.flip(data['B_down_err'])

I_down_neg = np.flip(data['I_down_neg'])
B_down_neg = np.flip(data['B_down_neg'])
B_down_neg_err = np.flip(data['B_down_neg_err'])

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

Bs = np.array([B_up, B_down, B_down_neg, B_up_neg])
Is = np.array([I_up, I_down, I_down_neg, I_up_neg])
Berrs = np.array([B_up_err, B_down_err, B_down_neg_err, B_up_neg_err])
Ierrs = np.array([I_up_err, I_down_err, I_down_neg_err, I_up_neg_err])

# interpolate and plot the date
curve_up = interpolate.make_interp_spline(I_up, B_up)
plt.plot(I_up, curve_up(I_up), color='blue',label='Interpolated graph')
plt.errorbar(I_up, B_up, B_up_err, I_up_err, label='Data', fmt='none', color='orange')

for i in range(len(Bs)-1):
    curve = interpolate.make_interp_spline(Is[i+1], Bs[i+1])
    plt.plot(Is[i+1], curve(Is[i+1]), color='blue')
    plt.errorbar(Is[i+1], Bs[i+1], Berrs[i+1], Ierrs[i+1], fmt='none', color='orange')
plt.xlabel('I [A]')
plt.ylabel('B [T]')
plt.title('The curve of the magnetic field strength (B) as a function of the current (I)')
plt.legend()
plt.show()


'''
# See what degree of polynomial is required
plt.errorbar(I_up, B_up, B_up_err, I_up_err, label='B(I)', fmt='none', color='orange')

for i in range(6):
    fit, resid = poly(I_up, B_up, i)
    y = 0
    for n in range(i+1):
        y += fit[n]* I_up**(n)
    print(resid)
    plt.plot(I_up, y, label='n='+str(i))

plt.legend()
plt.show()

print("---")

plt.errorbar(I_down, B_down, B_down_err, I_down_err, label='B(I)', fmt='none', color='orange')

for i in range(6):
    fit, resid = poly(I_down, B_down, i)
    y = 0
    for n in range(i+1):
        y += fit[n]* I_down**(n)
    print(resid)
    plt.plot(I_down, y, label='n='+str(i))

plt.legend()
plt.show()

print("---")

plt.errorbar(I_down_neg, B_down_neg, B_down_neg_err, I_down_neg_err, label='B(I)', fmt='none', color='orange')

for i in range(6):
    fit, resid = poly(I_down_neg, B_down_neg, i)
    y = 0
    for n in range(i+1):
        y += fit[n]* I_down_neg**(n)
    print(resid)
    plt.plot(I_down_neg, y, label='n='+str(i))

plt.legend()
plt.show()

print("---")

plt.errorbar(I_up_neg, B_up_neg, B_up_neg_err, I_up_neg_err, label='B(I)', fmt='none', color='orange')

for i in range(6):
    fit, resid = poly(I_up_neg, B_up_neg, i)
    y = 0
    for n in range(i+1):
        y += fit[n]* I_up_neg**(n)
    print(resid)
    plt.plot(I_up_neg, y, label='n='+str(i))

plt.legend()
plt.show()
'''

coeffs = open('coefficients_file.txt','w', encoding='utf-8')
coeffs.write('x0\tx1\tx2\tx3\tx4\tx5\n')

labels = ["B(I); I > 0, B' > 0", "B(I); I > 0, B' < 0", "B(I); I < 0, B' < 0", "B(I); I < 0, B' > 0"]

B_calc = np.empty_like(B_up)

for i in range(len(Bs)):
    x = np.linspace(Is[i][0], Is[i][-1], 1000)
    fit = poly(Is[i], Bs[i], 5)[0]
    y = 0
    for n in range(6):
        y += fit[n]* x**n
        coeffs.write(str(fit[n])+'\t')

    plt.plot(x,y, label=labels[i])
    coeffs.write('\n')
    print(f'B = {fit[0]} + {fit[1]}I + {fit[2]}I\u00B2 + {fit[3]}I\u00B3 + {fit[4]}I\u2074 + {fit[5]}I\u2075')

    # calculate the new values of B
    BI = fit[0] + fit[1]*Is[i] + fit[2]*Is[i]**2 + fit[3]*Is[i]**3 + fit[4]*Is[i]**4 + fit[5]*Is[i]**5
    B_calc = np.vstack((B_calc, BI))


plt.errorbar(I_up, B_up, B_up_err, I_up_err, label='Data', fmt='none', color='black')

for i in range(len(Bs)-1):
    plt.errorbar(Is[i+1], Bs[i+1], Berrs[i+1], Ierrs[i+1], fmt='none', color='black')


plt.xlabel('I [A]')
plt.ylabel('B [T]')
plt.title('The curve of the magnetic field strength (B) as a function of\nthe current (I) with the curves B(I) fit')
plt.legend()
plt.show()

