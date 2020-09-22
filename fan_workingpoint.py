import pandas as pd
import numpy as np
from database_creator import *
import matplotlib.pyplot as plt

Vdot_inst=np.array([374.1412,
555.213,
750.6173,
912.852,
1075.8731,
1156.4864,])
Psat_inst=np.array([
14.57571429,
44.43285714,
86.17285714,
127.2742857,
174.5642857,
203.69,
])

df_inst=pd.DataFrame(columns=['Flowrate - [m^3/h]','Static Pressure - [Pa]'],data=np.vstack((Vdot_inst,Psat_inst)).transpose())

fan_reference='9GV0824P1G03'
N_fans=3

database=import_database(database_path)

df_fan=fan_characteristic(database,fan_reference,'PWM',100)

df_sol=hydraulic_working_point(Vdot_inst,Psat_inst,df_fan['Vdot']*N_fans,df_fan['Psat'],plot='on')

