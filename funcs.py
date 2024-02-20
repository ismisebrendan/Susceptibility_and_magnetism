import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Round to the first significant figure of the uncertainty
def round_sig_fig_uncertainty(value, uncertainty):
    # check if numpy array or float
    if isinstance(value, np.ndarray):
        v_arr = np.array([])
        u_arr = np.array([])
        for i in range(len(value)):
            if uncertainty[i] == 0:
                v_arr = np.append(v_arr, value[i])
                u_arr = np.append(u_arr, uncertainty[i])
            else:
                value_out = np.round(value[i], int(np.abs(np.floor(np.log10(uncertainty[i])))))
                uncertainty_out = np.round(uncertainty[i], int(np.abs(np.floor(np.log10(uncertainty[i])))))

                v_arr = np.append(v_arr, value_out)
                u_arr = np.append(u_arr, uncertainty_out)
        return v_arr, u_arr
   
    elif isinstance(value, float):
        if uncertainty == 0:
            return value, uncertainty
        else:
            value_out = np.round(value, int(np.abs(np.floor(np.log10(uncertainty)))))
            uncertainty_out = np.round(uncertainty, int(np.abs(np.floor(np.log10(uncertainty)))))

            return value_out, uncertainty_out
    else:
        return value, uncertainty
    
    
# Plot and annotate a graph
def plot_fit(fit):
    slope, slope_err = round_sig_fig_uncertainty(fit.slope, fit.stderr)
    intercept, intercept_err = round_sig_fig_uncertainty(fit.intercept, fit.intercept_stderr)

    plt.plot([], [], ' ', label='R$^2$='+str(np.round(fit.rvalue, 6)))
    plt.plot([], [], ' ', label='$F_Z=Cm\chi B^2+a$')
    plt.plot([], [], ' ', label='$Cm\chi$ ='+str(slope)+r'$ \pm $'+str(slope_err)+' $Jm^{-1}T^{-2}$')
    plt.plot([], [], ' ', label='a='+str(intercept)+r'$ \pm $'+str(intercept_err)+' N')

# Plot and annotate a graph
def plot_fit4(fit):
    slope, slope_err = round_sig_fig_uncertainty(fit.slope, fit.stderr)
    intercept, intercept_err = round_sig_fig_uncertainty(fit.intercept, fit.intercept_stderr)

    plt.plot([], [], ' ', label='R$^2$='+str(np.round(fit.rvalue, 6)))
    plt.plot([], [], ' ', label='$F_Z=Cm\sigma B+a$')
    plt.plot([], [], ' ', label='$Cm\sigma$ ='+str(slope)+r'$ \pm $'+str(slope_err)+' $Jm^{-1}T^{-2}$')
    plt.plot([], [], ' ', label='a='+str(intercept)+r'$ \pm $'+str(intercept_err)+' N')

# An n-order polynomial
def poly(x, y, n):
    coefs = np.polynomial.polynomial.Polynomial.fit(x, y, n, full=True)[0].convert().coef
    resid = np.polynomial.polynomial.Polynomial.fit(x, y, n, full=True)[1][0]
    return coefs, resid

# quadratic model
def quad(p, x):
    # m = ax^2 + bx + c
    m = p[0]*x**2 + p[1]*x
    return m

# cubic model
def cubic(p, x):
    # m = ax^3 + bx^2 + cx
    m = p[0]*x**3 + p[1]*x**2 + p[2]*x
    return m

def sqr(p,x):
    # m = ax^2
    m = p*x**2
    return m

# merit function
def chi2(p, x, y, func):
    return np.sum((y - func(p,x))**2)

# Fitting function
def fitting(p, x, y, func, xerr, yerr, colour, label):

    # fit
    p_fit = opt.fmin(chi2, p, (x, y, func))

    chi_score = chi2(p_fit, x, y, func)

    print("for fit chi2 =", chi2(p_fit, x, y, func))

    plt.errorbar(x, y, yerr=yerr, xerr=xerr, label='Data', fmt='o', color=colour)
    plt.plot(x, func(p_fit,x), label=label, color=colour)

    return p_fit, chi_score
