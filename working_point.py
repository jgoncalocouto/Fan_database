import pandas as pd
import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt

#region Inputs:

'''Fan Restrictions'''
N_fan=11
fan_name='9GV0648P1H03_57'

'''Installation'''
type='Inlet'
entry_geometry='Concealed'
number_of_inlets='Single'
filter='G3'
area='Standard'
N_filter=3.5
fan_array_details='11 fans of φ60 and W=38'
#fan_array_details='3 fans of φ160 and W=51'

#endregion


#region Fan
database_path = 'fan_database/'
data_xls = pd.ExcelFile(database_path + 'sanyo_denki_flow.xlsx')
df_fan = data_xls.parse(fan_name)
#endregion


list=pd.DataFrame({'rpm': df_fan['Rotational Speed - [rpm]'].unique(), 'flowrate':np.nan, 'static pressure':np.nan})

#region Installation
database2_path = 'installation_database/'
data2_xls=pd.ExcelFile(database2_path + '10_03_2020_installation_database.xlsx')
df_installation = data2_xls.parse('Pressure Loss Curves')

df_inst_selected=df_installation[
    (df_installation['Type']==type) &
    (df_installation['Entry Geometry']==entry_geometry) &
    (df_installation['Number of inlets']==number_of_inlets) &
    (df_installation['Filter']==filter) &
    (df_installation['Cross-sectional Area']==area) &
    (df_installation['N_filter']==N_filter) &
    (df_installation['Fan Array Details']==fan_array_details)
]
a2=float(df_inst_selected['a2'])
a1=float(df_inst_selected['a1'])
a0=float(df_inst_selected['a0'])
#endregion



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
        plt.plot(Vdot, inst_psat, label=df_inst_selected['Number'].values[0])

    plt.plot(Vdot, P_fan, label=['N_fans = '+str(N_fan)+'; RPM='+str(list['rpm'].loc[ind])])



print(list)

plt.xlabel('Air Flowrate - [m^3/h]')
plt.ylabel('Installation Pressure Loss - [Pa]')
plt.title('Working point prediction')
plt.legend()