import pandas as pd
import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt

#region Inputs:
'''Fan Restrictions'''
N_fans_max=50
'''Flowrate'''
flow_acceptance_criteria=100
flow_setpoint=900

'''Installation'''
type='Inlet'
entry_geometry='Concealed'
number_of_inlets='Single'
filter='G4'
area='Standard'
N_filter=3.5
#endregion

#region Fan
database_path = 'fan_database/'
data_xls = pd.ExcelFile(database_path + 'sanyo_denki_flow.xlsx')
list_of_fans=data_xls.sheet_names
#endregion

#region Installation
database2_path = 'installation_database/'
data2_xls=pd.ExcelFile(database2_path + '04_03_2020_installation_database.xlsx')
df_installation = data2_xls.parse('Pressure Loss Curves')

df_inst_selected=df_installation[
    (df_installation['Type']==type) &
    (df_installation['Entry Geometry']==entry_geometry) &
    (df_installation['Number of inlets']==number_of_inlets) &
    (df_installation['Filter']==filter) &
    (df_installation['Cross-sectional Area']==area) &
    (df_installation['N_filter']==N_filter)
]
a2=float(df_inst_selected['a2'])
a1=float(df_inst_selected['a1'])
a0=float(df_inst_selected['a0'])
#endregion

'''Data frame initialization'''
df_result=pd.DataFrame(data_xls.parse(list_of_fans[0]).iloc[0]).transpose()
df_result['Flowrate - [m^3/min]'] = 0
df_result['Static Pressure - [Pa]'] = 0
df_result["Number of fans selected"] = 0
df_result["Installation Code"] = df_inst_selected['Number']

fan_idx = 0
for fan in list_of_fans:
    #fan='9CRA0612P0S001'
    df_fan = data_xls.parse(fan)
    aux=np.full([50, 4], np.nan)
    for i in range(1,N_fans_max+1):
        #i=11
        fan_flow = df_fan[df_fan['PWM - [%]'] == 100]['Flowrate - [m^3/min]'] * 60 * i
        fan_psat = df_fan[df_fan['PWM - [%]'] == 100]['Static Pressure - [Pa]']
        fan_p_iter = interpolate.interp1d(fan_flow, fan_psat, kind='quadratic')
        Vdot = np.arange(min(fan_flow), max(fan_flow), 1)
        P_fan = fan_p_iter(Vdot)
        inst_psat = a2 * Vdot ** 2 + a1 * Vdot + a0
        idx = np.argwhere(np.diff(np.sign(P_fan - inst_psat))).flatten()
        if idx.size==0:
            aux[i-1]=[np.nan,np.nan,np.nan,i]
        else:
            Vdot_iter=Vdot[idx]
            Psat_iter=P_fan[idx]
            aux[i-1] = [Vdot_iter , Psat_iter, (Vdot_iter - flow_setpoint), i]

    idx_sol=np.nanargmin(abs(aux[:, 2]))
    if (aux[idx_sol,0]<flow_setpoint and aux[idx_sol,3]<50):
        idx_sol+=1
    Vdot_iter = aux[idx_sol, 0]
    Psat_iter = aux[idx_sol, 1]
    N_fans_iter = aux[idx_sol, 3]


    df_i = pd.DataFrame(data_xls.parse(fan).iloc[0]).transpose()
    df_i['Flowrate - [m^3/min]'].loc[0] = Vdot_iter
    df_i['Static Pressure - [Pa]'].loc[0]=Psat_iter
    df_i["Number of fans selected"]= N_fans_iter
    df_i["Installation Code"]=str(df_inst_selected['Number'].values[0])
    df_result.loc[fan_idx]=df_i.loc[0]
    df_result.append(df_result.loc[0])
    fan_idx+=1

#df_result.to_csv(database2_path + str(df_inst_selected['Number'].values[0])+'_'+'04_03_2020_installation_database.csv')