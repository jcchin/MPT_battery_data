import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import odeint

import time
st = time.time()


T_bp = np.array([0., 20., 30., 45., 60.])
SOC_bp = np.array( [-0.1, 0.        , 0.03333333, 0.06666667, 0.1       , 0.13333333, 0.16666667,
 0.2       , 0.23333333, 0.26666667, 0.3       , 0.33333333, 0.36666667,
 0.4       , 0.43333333, 0.46666667, 0.5       , 0.53333333, 0.56666667,
 0.6       , 0.63333333, 0.66666667, 0.7       , 0.73333333, 0.76666667,
 0.8       , 0.83333333, 0.86666667, 0.9       , 0.93333333, 0.96666667,
 1.        ] )
tU_oc = np.array([ [2.92334783, 2.92334783,3.00653623,3.08972464,3.17291304,3.23989855,3.31010145,
 3.3803913 ,3.44033333,3.49033333,3.52169565,3.54391304,3.58695652,
 3.62095652,3.65437681,3.68604348,3.72430435,3.75531884,3.79102899,
 3.82030435,3.84181159,3.86124638,3.88921739,3.91686957,3.96223188,
 4.00169565,4.04117391,4.06849275,4.07573913,4.08571014,4.10571014,
 4.161     ] ,  
 [2.99293893,2.99293893,3.05400763,3.11507634,3.17614504,3.23506616,3.30371247,
 3.37521374,3.43605852,3.48697455,3.5200229 ,3.54251908,3.58374046,
 3.6329313 ,3.67379644,3.70287532,3.73784733,3.76526463,3.79174809,
 3.81922901,3.84108142,3.87212214,3.90738931,3.93615267,3.98113995,
 4.02093893,4.04504071,4.07114758,4.07583969,4.08371501,4.10560814,
 4.161     ] ,  
 [2.84084639,2.84084639,2.98428484,3.1050295 ,3.19464496,3.25566531,3.309059  ,
 3.37185148,3.43473652,3.49059613,3.51955239,3.541353  ,3.58558494,
 3.62641607,3.6708881 ,3.70814547,3.7392177 ,3.76822075,3.79592981,
 3.82260427,3.84986368,3.88146592,3.91739674,3.94798779,3.98188403,
 4.02274568,4.05623296,4.06830824,4.07468871,4.08175788,4.10853306,
 4.153     ] ,  
 [2.81925101,2.81925101,2.97410931,3.09861134,3.18674899,3.24142105,3.29678138,
 3.35963563,3.42195951,3.47637247,3.51383806,3.54319838,3.59076923,
 3.61940891,3.65574089,3.7067004 ,3.74153441,3.77023887,3.79773684,
 3.82421053,3.85139271,3.88311336,3.91906478,3.94918219,3.98310931,
 4.02401215,4.05611741,4.07036842,4.07774494,4.08190283,4.10867206,
 4.153     ],
 [2.79852227,2.79852227,2.95540486,3.07335223,3.17051822,3.228     ,3.29222672,
 3.34192308,3.39768016,3.46263158,3.49862348,3.53679757,3.59007692,
 3.61315789,3.65250202,3.70067206,3.74025911,3.77023887,3.79773684,
 3.82335628,3.84432794,3.87774899,3.91851822,3.94918219,3.98310931,
 4.02401215,4.05611741,4.06846559,4.07574494,4.08185425,4.10867206,
 4.151     ] ])
tC_Th = np.array([ [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.] ,  
 [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.] ,  
 [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.] ,  
 [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.] ,
 [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
 2000.,2000.,2000.,2000.,2000.,2000.,2000.] , ])

tR_Th = np.array(
[ [0.09      ,0.09      ,0.09      ,0.09      ,0.09      ,0.07130435,0.06      ,
 0.06      ,0.06      ,0.06      ,0.06      ,0.06      ,0.06      ,
 0.08217391,0.07492754,0.07      ,0.07      ,0.07      ,0.07      ,
 0.07      ,0.05318841,0.04144928,0.0573913 ,0.06884058,0.07      ,
 0.07456522,0.075     ,0.075     ,0.05586957,0.055     ,0.04021739,
 0.04      ] ,  
 [0.08534351,0.08534351,0.07516539,0.06498728,0.05480916,0.04838931,0.04589059,
 0.045     ,0.045     ,0.04195929,0.03937405,0.03642494,0.035     ,
 0.035     ,0.03848601,0.05430025,0.04534351,0.03624682,0.03115776,
 0.03      ,0.03      ,0.03      ,0.03839695,0.04      ,0.04      ,
 0.04      ,0.03089059,0.03      ,0.03      ,0.02807125,0.02505344,
 0.02      ] ,  
 [0.0677823 ,0.0677823 ,0.05252289,0.045     ,0.045     ,0.045     ,0.045     ,
 0.04207528,0.04      ,0.03690234,0.035     ,0.0317294 ,0.02798576,
 0.027     ,0.025588  ,0.025     ,0.02129705,0.02      ,0.02      ,
 0.04377416,0.04190234,0.04      ,0.04      ,0.04      ,0.03121058,
 0.02820753,0.028     ,0.02055341,0.02      ,0.02      ,0.02      ,
 0.001     ] ,  
 [0.06728745,0.06728745,0.04704453,0.04      ,0.04      ,0.04      ,0.04      ,
 0.04      ,0.04      ,0.03267206,0.03      ,0.03      ,0.03      ,
 0.03      ,0.02603239,0.025     ,0.02091093,0.02      ,0.02      ,
 0.04562753,0.04133603,0.04      ,0.04      ,0.04      ,0.0308502 ,
 0.02814575,0.028     ,0.02038866,0.02      ,0.02      ,0.02      ,
 0.001     ],
[0.06728745,0.06728745,0.04704453,0.04      ,0.04      ,0.04      ,0.04      ,
 0.04      ,0.04      ,0.03267206,0.03      ,0.03      ,0.03      ,
 0.03      ,0.02603239,0.025     ,0.02091093,0.02      ,0.02      ,
 0.02      ,0.02433198,0.03378543,0.035     ,0.035     ,0.0304251 ,
 0.02536437,0.025     ,0.02024291,0.02      ,0.02      ,0.02      ,
 0.001     ]])

tR_0 = np.array([ [0.2473913 ,0.2473913 ,0.20681159,0.16623188,0.12565217,0.09753623,0.08362319,
 0.08      ,0.07666667,0.0715942 ,0.07      ,0.0415942 ,0.05681159,
 0.067     ,0.067     ,0.067     ,0.067     ,0.067     ,0.06537681,
 0.065     ,0.065     ,0.065     ,0.065     ,0.065     ,0.065     ,
 0.065     ,0.065     ,0.065     ,0.065     ,0.065     ,0.065     ,
 0.065     ] ,  
 [0.08801527,0.08801527,0.07274809,0.05748092,0.04221374,0.03231552,0.02722646,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.025     ] ,  
 [0.0677823 ,0.0677823 ,0.05252289,0.03726348,0.02733469,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.01311292,0.01809766,0.02      ,0.02      ,0.02430824,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.03      ] ,  
 [0.06546559,0.06546559,0.0502834 ,0.03510121,0.02663968,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.01218623,0.01      ,0.01      ,0.01890688,0.02451417,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.03      ],
 [0.06546559,0.06546559,0.0502834 ,0.03510121,0.02663968,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.02072874,0.02      ,0.02      ,0.02      ,0.02451417,0.025     ,
 0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,0.025     ,
 0.03      ]])

X,Y = np.meshgrid(SOC_bp, T_bp);

SMOOTHING = 0.01 # more is more smooth, less true to the data
U_oc_interp = RectBivariateSpline(T_bp, SOC_bp, tU_oc, s=SMOOTHING) # % need Deg C
C_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tC_Th, s=SMOOTHING) # % need Deg C
R_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tR_Th, s=SMOOTHING) # % need Deg C
R_0_interp = RectBivariateSpline(T_bp, SOC_bp, tR_0, s=SMOOTHING) # % need Deg C

with open('Data Archive/X-57_L1_004.csv', 'r') as data_file:

    header = data_file.readline().strip().split(",")

    time_col = header.index("Test_Time(s)")
    current_col = header.index("Current(A)")
    voltage_col = header.index("Voltage(V)")
    temp_col = header.index("Temperature (C)_1")

    data = []
    for line in data_file:
        row = line.strip().split(",")
        data.append([float(x) for x in (row[time_col], row[current_col], row[voltage_col], row[temp_col])])

    start_idx = 76
    end_idx = 15100

    test_data = np.array(data)[start_idx:end_idx]
    test_data[:,0] -= test_data[0,0] # set t= 0 to start index
    test_data[:,1] *= -1 # flip sign on current


current_interp = interp1d(test_data[:,0], test_data[:,1], bounds_error=False, kind='linear')
temp_interp = interp1d(test_data[:,0], test_data[:,3], bounds_error=False, kind='linear')


def ode_func(y, t):

    # print(y, t)
    SOC = y[0]
    U_Th = y[1]
    T_batt = y[2]#temp_interp(t);

    Q_max = 3.0  #2.85 bottoms out, 3.0 doesn't make it to end of table
    mass_cell = 0.045
    Cp_cell = 1020
    eff_cell = 0.95
    Pack_Loss = 1.0

    if t>0.:
        I_Li = current_interp(t)
    else:
        I_Li = 0.

    U_oc = U_oc_interp(T_batt, SOC)
    C_Th = C_Th_interp(T_batt, SOC)
    R_Th = R_Th_interp(T_batt, SOC)
    R_0 = R_0_interp(T_batt, SOC)


    dXdt_SOC = -I_Li / (3600.0 * Q_max);
    dXdt_U_Th = -U_Th / (R_Th * C_Th) + I_Li / (C_Th);
    dXdt_T_batt = (I_Li**2 *(R_0+R_Th)) / (mass_cell*Cp_cell)
    dXdt_T_batt = dXdt_T_batt - (.06*(y[2]-19.) / (mass_cell*Cp_cell))
    # U_L = U_oc - U_Th - (I_Li * R_0);\

    return [dXdt_SOC, dXdt_U_Th, dXdt_T_batt]


def compute_other_vars(y, t):
    SOC = y[:,0]
    U_Th = y[:,1]
    T_batt = temp_interp(t)

    I_Li = current_interp(t)


    U_oc_vals = []
    R_0_vals = []
    for i in range(len(SOC)):
        U_oc = U_oc_interp(T_batt[i], SOC[i])[0]
        C_Th = C_Th_interp(T_batt[i], SOC[i])[0]
        R_Th = R_Th_interp(T_batt[i], SOC[i])[0]
        R_0 = R_0_interp(T_batt[i], SOC[i])[0]
        
        U_oc_vals.append(U_oc[0])
        R_0_vals.append(R_0[0])

    U_oc = np.array(U_oc_vals)
    R_0 = np.array(R_0_vals)


    U_L = U_oc - U_Th - (I_Li * R_0)

    return U_L



sim_states = odeint(ode_func, y0=[1, 0, 19], t=test_data[:,0], hmax=4)# , atol=1e-12)
sim_data = compute_other_vars(sim_states, test_data[:,0])

print('sim time', time.time() - st)
print(min(sim_states[:,0]))

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.set_ylabel('Voltage')
ax1.plot(test_data[:,0], test_data[:,2])
ax1.plot(test_data[:,0], sim_data, linewidth=1.)

# Thermal Plot
ax2.set_ylabel('Temperature (C)')
ax2.plot(test_data[:,0], test_data[:,3])
ax2.plot(test_data[:,0], sim_states[:,2], linewidth=1.)
ax2.set_xlabel('Time (s)')

plt.show()
