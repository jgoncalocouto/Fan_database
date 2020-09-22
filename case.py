import pandas as pd
import numpy as np
from database_creator import *
import matplotlib.pyplot as plt

Vdot=np.array([374.1412,
555.213,
750.6173,
912.852,
1075.8731,
1156.4864,])
Psat=np.array([
14.57571429,
44.43285714,
86.17285714,
127.2742857,
174.5642857,
203.69,
])

df_inst=pd.DataFrame(columns=['Flowrate - [m^3/h]','Static Pressure - [Pa]'],data=np.vstack((Vdot,Psat)).transpose())

fan_reference='9GV0824P1G03'

df_database = import_fan_data('fan_database/','sanyo_denki_flow.xlsx')
df_database["Flowrate - [m^3/h]"]=df_database["Flowrate - [m^3/min]"]*60


df_fan=fan_characteristic(df_database,fan_reference,'PWM',100,'Vdot vs Psat')

