import pandas as pd
import numpy as np
from database_creator import *
import matplotlib.pyplot as plt

Vdot_inst=np.array([
    356.6244,
    532.7254,
    721.7369,
    878.1878,
    1036.2305,
    1115.3166,
])
Psat_inst=np.array([
    16.88857143,
    53.50571429,
    94.56714286,
    139.5128571,
    192.4642857,
    222.0757143,
])

df_inst=pd.DataFrame(columns=['Flowrate - [m^3/h]','Static Pressure - [Pa]'],data=np.vstack((Vdot_inst,Psat_inst)).transpose())

fan_reference='9GT0924P1M001'
N_fans=8

database=import_database(database_path)

df_fan=fan_characteristic(database,fan_reference,'PWM',100)

df_sol=hydraulic_working_point(Vdot_inst,Psat_inst,df_fan['Vdot']*N_fans,df_fan['Psat'],plot='on')

