import pandas as pd
import numpy as np
import pprint
from scipy import interpolate
import math
import matplotlib.pyplot as plt
import sys


#region Inputs:

'''Fan Restrictions'''
N_fan=10
fan_name='9GA0824P1H61'

'''Installation'''
type='Inlet'
entry_geometry='Concealed'
number_of_inlets='Single'
filter='G3'
area='Standard'
N_filter=1.0
fan_array_details='14 fans of φ60 and W=76'
#fan_array_details='3 fans of φ160 and W=51'

#endregion


#region Fan
database_path = 'fan_database/'
data_xls = pd.ExcelFile(database_path + 'sanyo_denki_flow.xlsx')
df_fan = data_xls.parse(fan_name)
#endregion


list=pd.DataFrame({'rpm': df_fan['Rotational Speed - [rpm]'].unique(), 'pwm': df_fan['PWM - [%]'].unique(), 'flowrate':np.nan, 'static pressure':np.nan})

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

try:
    data3_xls = pd.ExcelFile(database_path + 'sanyo_denki_sound.xlsx')
    df_spl = data3_xls.parse(fan_name)
except:
    print('The fan selected does not have sound level data')

df_spl['Flowrate - [m^3/h]']=np.nan
df_spl['Static Pressure - [Pa]']=np.nan

#region Sound Level




for i in range(len(df_spl['PWM - [%]'])):
    rpm_iter = interpolate.interp1d(list['pwm'], list['rpm'], kind='linear')
    try:
        rpm_i=rpm_iter(df_spl['PWM - [%]'].loc[i])
    except:
        continue
    ind=i

    fan_flow=df_fan[df_fan['Rotational Speed - [rpm]'] == list['rpm'].loc[0]]['Flowrate - [m^3/min]']*60*N_fan*(rpm_i/list['rpm'].loc[0])
    fan_psat=df_fan[df_fan['Rotational Speed - [rpm]'] == list['rpm'].loc[0]]['Static Pressure - [Pa]']*((rpm_i**2)/(list['rpm'].loc[0]**2))
    fan_p_iter = interpolate.interp1d(fan_flow, fan_psat, kind='quadratic')
    Vdot=np.arange(min(fan_flow),max(fan_flow),1)
    P_fan=fan_p_iter(Vdot)

    inst_psat=a2*Vdot**2+a1*Vdot+a0

    idx = np.argwhere(np.diff(np.sign(P_fan - inst_psat))).flatten()
    df_spl['Flowrate - [m^3/h]'][ind]=Vdot[idx]
    df_spl['Static Pressure - [Pa]'][ind]=inst_psat[idx]

    plt.figure(1)
    plt.plot(Vdot, P_fan, label=['N_fans = '+str(N_fan)+'; RPM='+str(rpm_i.round())])

    if ind==(len(df_spl['PWM - [%]'])-1):
        plt.plot(Vdot, inst_psat, label=df_inst_selected['Number'].values[0])




print([df_spl['Flowrate - [m^3/h]'], df_spl['Static Pressure - [Pa]'] , df_spl['Sound Level - [db-A]']])

plt.xlabel('Air Flowrate - [m^3/h]')
plt.ylabel('Installation Pressure Loss - [Pa]')
plt.title('Working point prediction')
plt.legend()

plt.figure(2)
plt.plot(df_spl['Flowrate - [m^3/h]'], df_spl['Sound Level - [db-A]'])
plt.xlabel('Air Flowrate - [m^3/h]')
plt.ylabel('Sound Level (measured at 1m from the fan inlet for a fan with a free outlet) - [dB-A]')
plt.title('Sound Level Prediction for '+'Fan = '+fan_name+'; with N= '+str(N_fan))


spl_iter=interpolate.interp1d(df_spl['Flowrate - [m^3/h]'], df_spl['Sound Level - [db-A]'], kind='linear')


plt.scatter(np.array([800,900,max(df_spl['Flowrate - [m^3/h]'])]),spl_iter([800,900,max(df_spl['Flowrate - [m^3/h]'])]))

for point in np.array([800,900,max(df_spl['Flowrate - [m^3/h]'])]):
    plt.text(point*1,spl_iter(point)*0.9,str(spl_iter(point).round()))
