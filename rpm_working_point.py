import pandas as pd
import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt

#region Fan
database_path = 'fan_database/'
data_xls = pd.ExcelFile(database_path + 'sanyo_denki_flow.xlsx')
df_fan = data_xls.parse('9CRA0612P0S001')
N_fan=11

#endregion


list=pd.DataFrame({'rpm': df_fan['Rotational Speed - [rpm]'].unique(), 'flowrate':np.nan, 'static pressure':np.nan})

#region Installation
cases=pd.DataFrame({'case_designation': ['template'], 'a2':np.nan, 'a1':np.nan, 'a0':np.nan})
cases=cases.append({
    'case_designation': 'Single Inlet Section, Filter=G4, N_filter=3.5',
    'a2': 0.0004,
    'a1': 0.2165,
    'a0': -3.8987
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Single Inlet Section, Filter=G4, N_filter=1.0',
    'a2': 0.0003,
    'a1': 0.0461,
    'a0': 13.985
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Triple Inlet Section (Front), Filter=G4, N_filter=3.5',
    'a2': 0.0005,
    'a1': 0.199,
    'a0': -0.3548
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Triple Inlet Section (Cup), Filter=G4, N_filter=3.5',
    'a2': 0.0151,
    'a1': 0.8439,
    'a0': 1.9692
},ignore_index=True)


cases=cases.append({
    'case_designation': 'Triple Inlet Section (Front), Filter=G4, N_filter=1.0',
    'a2': 0.0004,
    'a1': -0.0551,
    'a0': 31.547
},ignore_index=True)


cases=cases.append({
    'case_designation': 'Triple Inlet Section (Cup), Filter=G4, N_filter=1.0',
    'a2': 0.0074,
    'a1': 0.2247,
    'a0': 21.541
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Single Outlet Section, Filter=G4, N_filter=3.5',
    'a2': 0.0003,
    'a1': 0.1675,
    'a0': - 1.195
},ignore_index=True)


cases=cases.append({
    'case_designation': 'Single Outlet Section, Filter=G4, N_filter=1.0',
    'a2': 0.0002,
    'a1': 0.048,
    'a0': -1.0166
},ignore_index=True)


cases=cases.append({
    'case_designation': 'Single Inlet Section, Filter=G3, N_filter=3.5',
    'a2': 0.0003,
    'a1': 0.0658,
    'a0': 7.1203
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Single Outlet Section, Filter=G3, N_filter=3.5',
    'a2': 0.0003,
    'a1': 0.0344,
    'a0': 4.7606
},ignore_index=True)


cases=cases.append({
    'case_designation': 'Triple Outlet Section (Cup), Filter=G3, N_filter=3.5',
    'a2': 0.0084,
    'a1': 0.2078,
    'a0': 22.117
},ignore_index=True)


cases=cases.append({
    'case_designation': 'Triple Outlet Section (Front), Filter=G3, N_filter=3.5',
    'a2': 0.0005,
    'a1': -0.0593,
    'a0': 32.51
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Triple Inlet Section (Front), Filter=G4, Grille, N_filter=3.5',
    'a2': 0.0005,
    'a1': 0.0775,
    'a0': 29.252
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Single Inlet Section, Filter=G3, Grille, N_filter=3.5',
    'a2': 0.0002,
    'a1': 0.0591,
    'a0': 7.0179
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Triple Inlet Section (Front), Filter=G3, Grille, N_filter=3.5',
    'a2': 0.0004,
    'a1':-0.0582,
    'a0': 30.497
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Single Outlet Section, Filter=G4, Grille, N_filter=3.5',
    'a2': 0.0002,
    'a1': 0.1456,
    'a0': 5.5894
},ignore_index=True)

cases=cases.append({
    'case_designation': 'Single Outlet Section, Filter=G3, Grille, N_filter=3.5',
    'a2': 0.0002,
    'a1': 0.0391,
    'a0': 4.1146
},ignore_index=True)





#endregion

selection='Single Inlet Section, Filter=G3, N_filter=3.5'

a2=float(cases.loc[cases['case_designation']==selection]['a2'])
a1=float(cases.loc[cases['case_designation']==selection]['a1'])
a0=float(cases.loc[cases['case_designation']==selection]['a0'])


for i in range(len(list['rpm'])):
    ind=i

    fan_flow=df_fan[df_fan['Rotational Speed - [rpm]'] == list['rpm'].loc[ind]]['Flowrate - [m^3/min]']*60*N_fan
    fan_psat=df_fan[df_fan['Rotational Speed - [rpm]'] == list['rpm'].loc[ind]]['Static Pressure - [Pa]']
    fan_p_iter = interpolate.interp1d(fan_flow, fan_psat, kind='quadratic')
    Vdot=np.arange(min(fan_flow),max(fan_flow),1)
    P_fan=fan_p_iter(Vdot)

    inst_psat=a2*Vdot**2+a1*Vdot+a0

    idx = np.argwhere(np.diff(np.sign(P_fan - inst_psat))).flatten()
    list['flowrate'][ind]=Vdot[idx]
    list['static pressure'][ind]=inst_psat[idx]

    if ind==0:
        plt.plot(Vdot, inst_psat, label=selection)

    plt.plot(Vdot, P_fan, label=['N_fans = '+str(N_fan)+'; RPM='+str(list['rpm'].loc[ind])])



print(list)



plt.xlabel('Air Flowrate - [m^3/h]')
plt.ylabel('Installation Pressure Loss - [Pa]')
plt.title('Working point prediction')
plt.legend()