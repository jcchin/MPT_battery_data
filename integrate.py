import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import odeint

import time
st = time.time()


# X57_MPT_1C_lot1_027_20C.csv
# X57_MPT_1C_lot1_028_20C.csv
# X57_MPT_1C_lot1_029_20C.csv

# X57_MPT_1C_lot1_030_0C.csv
# X57_MPT_1C_lot1_040_30C
# X57_MPT_1C_lot1_043_45C

with open('Data Archive/X57_MPT_1C_lot1_027_20C.csv', 'r') as data_file:

    header = data_file.readline().strip().split(",")

    time_col = header.index("Test_Time(s)")
    current_col = header.index("Current(A)")
    voltage_col = header.index("Voltage(V)")


    data = []
    for line in data_file:
        row = line.strip().split(",")
        data.append([float(x) for x in (row[time_col], row[current_col], row[voltage_col])])

    start_idx = 71
    end_idx = 14000

    test_data = np.array(data)[start_idx:end_idx]
    test_data[:,0] -= test_data[0,0]
    test_data[:,1] *= -1


current_interp = interp1d(test_data[:,0], test_data[:,1], bounds_error=False, kind='linear')


T_bp = [20., 40.] # % Deg C
SOC_bp = np.linspace(0.083,1,29) # %[0., 0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.953, 1.] #%  # SOC break points

tU_oc = [[3.145, 3.205, 3.261, 3.338, 3.403, 3.459, 3.505, 3.529, 3.550, 3.601, 3.648, 3.685, 3.710, 3.748, 3.771, 3.798, 3.825, 3.845, 3.878, 3.913, 3.940, 3.987, 4.025, 4.047, 4.073, 4.076, 4.084, 4.106, 4.161],
         [3.145, 3.205, 3.261, 3.338, 3.403, 3.459, 3.505, 3.529, 3.550, 3.601, 3.648, 3.685, 3.710, 3.748, 3.771, 3.798, 3.825, 3.845, 3.878, 3.913, 3.940, 3.987, 4.025, 4.047, 4.073, 4.076, 4.084, 4.106, 4.161]]  # volts, Open Circuit Voltage

tC_Th = [[2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.],
         [2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.]]  # farads

tR_Th=  [[0.060, 0.050, 0.047, 0.045, 0.045, 0.045, 0.040, 0.039, 0.035, 0.035, 0.035, 0.040, 0.060, 0.040, 0.035, 0.030, 0.030, 0.030, 0.030, 0.040, 0.040, 0.040, 0.040, 0.030, 0.030, 0.030, 0.028, 0.025, 0.020],
         [0.060, 0.045, 0.045, 0.045, 0.045, 0.045, 0.040, 0.039, 0.035, 0.035, 0.035, 0.040, 0.060, 0.040, 0.035, 0.030, 0.030, 0.030, 0.030, 0.040, 0.040, 0.040, 0.040, 0.030, 0.030, 0.030, 0.028, 0.025, 0.020]] # ohm  Ohmic Losses

tR_0 =  [[0.050, 0.035, 0.030, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025],
         [0.050, 0.035, 0.030, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]]  # ohm                           #                                                                                                        
    
X,Y = np.meshgrid(SOC_bp, T_bp);
#print(X.shape,Y.shape,len(tU_oc[0]))
U_oc_interp = interp2d(X, Y, tU_oc, kind='linear') # % need Deg C
C_Th_interp = interp2d(X, Y, tC_Th, kind='linear')
R_Th_interp = interp2d(X, Y, tR_Th, kind='linear')
R_0_interp = interp2d(X, Y, tR_0, kind='linear')


def ode_func(y, t):

    # print(y, t)
    SOC = y[0]
    U_Th = y[1]

    T_batt = 20;
    Q_max = 3.0;  #2.85 bottoms out, 3.0 doesn't make it to end of table
    mass_cell = 0.045;
    Cp_cell = 1020;
    eff_cell = 0.95;
    Pack_Loss = 1.0;

    I_Li = current_interp(t)

    U_oc = U_oc_interp(SOC, T_batt)
    C_Th = C_Th_interp(SOC, T_batt)
    R_Th = R_Th_interp(SOC, T_batt)
    R_0 = R_0_interp(SOC, T_batt)

    dXdt_SOC = -I_Li / (3600.0 * Q_max);
    dXdt_U_Th = -U_Th / (R_Th * C_Th) + I_Li / (C_Th);
    # U_L = U_oc - U_Th - (I_Li * R_0);\

    return [dXdt_SOC, dXdt_U_Th]


def compute_other_vars(y, t):
    SOC = y[:,0]
    U_Th = y[:,1]
    T_batt = 20 

    I_Li = current_interp(t)

    U_oc_vals = []
    R_0_vals = []
    for i in range(len(SOC)):
        U_oc = U_oc_interp(SOC[i], T_batt)
        C_Th = C_Th_interp(SOC[i], T_batt)
        R_Th = R_Th_interp(SOC[i], T_batt)
        R_0 = R_0_interp(SOC[i], T_batt)
        U_oc_vals.append(U_oc[0])
        R_0_vals.append(R_0[0])

    U_oc = np.array(U_oc_vals)
    R_0 = np.array(R_0_vals)

    U_L = U_oc - U_Th - (I_Li * R_0)

    return U_L

sim_states = odeint(ode_func, y0=[1, 0], t=test_data[:,0], hmax=4)# , atol=1e-12)
sim_data = compute_other_vars(sim_states, test_data[:,0])

print('sim time', time.time() - st)
print(min(sim_states[:,0]))

fig, ax = plt.subplots()

ax.plot(test_data[:,0], test_data[:,2])
ax.plot(test_data[:,0], sim_data, linewidth=1.)

plt.show()
