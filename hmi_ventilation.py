import pandas as pd
import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt

#region Inputs:
'''Fan Restrictions'''
N_fan=4
fan_name='FFB0824HHE'

#endregion


#region Fan
database_path = 'fan_database/'
data_xls = pd.ExcelFile(database_path + 'sanyo_denki_flow.xlsx')
df_fan = data_xls.parse(fan_name)
#endregion


list=pd.DataFrame({'rpm': df_fan['Rotational Speed - [rpm]'].unique(), 'flowrate':np.nan, 'static pressure':np.nan})

#region Installation
inst_flow=[
169.901,
203.8812,
246.35645,
326.20992
]

inst_p=[
23,
34,
48,
88
]


#endregion



for i in range(len(list['rpm'])):
    ind=i
    fan_flow=df_fan[df_fan['Rotational Speed - [rpm]'] == list['rpm'].loc[ind]]['Flowrate - [m^3/min]']*60*N_fan
    fan_psat=df_fan[df_fan['Rotational Speed - [rpm]'] == list['rpm'].loc[ind]]['Static Pressure - [Pa]']*2
    fan_p_iter = interpolate.interp1d(fan_flow, fan_psat, kind='quadratic')
    Vdot=np.arange(min(fan_flow),max(fan_flow),1)
    P_fan=fan_p_iter(Vdot)

    inst_p_iter = interpolate.InterpolatedUnivariateSpline(inst_flow, inst_p, k=2)

    inst_psat=inst_p_iter(Vdot)

    idx = np.argwhere(np.diff(np.sign(P_fan - inst_psat))).flatten()
    list['flowrate'][ind]=Vdot[idx]
    list['static pressure'][ind]=inst_psat[idx]

    if ind==0:
        plt.plot(Vdot, inst_psat, label='HMI Ventilation')

    plt.plot(Vdot, P_fan, label=['N_fans = '+str(N_fan*2)+' (4 Inlet / 4 Outlet)'+'; RPM='+str(list['rpm'].loc[ind])])



print(list)

plt.xlabel('Air Flowrate - [m^3/h]')
plt.ylabel('Installation Pressure Loss - [Pa]')
plt.title('Working point prediction')
plt.legend()