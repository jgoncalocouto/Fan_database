from database_creator import *

def PWM_predictor(ref,N,database,Vdot_target,Vdot_inst,Psat_inst,SPL_othersources):
    df_database = database['database']
    df_flow = database['flow']
    df_rpm = database['rpm']
    df_sound = database['sound']


    df_fan = df_flow[df_flow['Model'] == ref]
    PWM_min, PWM_max = min(df_fan['PWM - [%]'].unique()), max(df_fan['PWM - [%]'].unique())

    Vdot=0;
    PWM=PWM_min-1

    while Vdot < Vdot_target and PWM < PWM_max:
        PWM+=1
        df_flow_i=fan_characteristic(database,ref,'PWM',PWM)
        df_flow_i=df_flow_i.sort_values(by='Vdot',ascending=True)
        df_flow_i['Vdot']=df_flow_i['Vdot']*N

        Vdot,Psat=hydraulic_working_point(Vdot_inst, Psat_inst, df_flow_i['Vdot'], df_flow_i['Psat'])

        df_sound_fan=fan_sound(database,ref)

        f_sound=interpolate.interp1d(df_sound_fan['PWM'],df_sound_fan['SPL'],fill_value='extrapolate')
        SPL_fan=f_sound(PWM)
        SPL=10*math.log10(2*N*10**(SPL_fan/10)+10**(SPL_othersources/10))

        f_error_up=interpolate.interp1d(df_sound_fan['PWM'],df_sound_fan['SPL_error_up'],fill_value='extrapolate')
        SPL_error_up = f_error_up(PWM)
        SPL_up = 10 * math.log10(2 * N * 10 ** ((SPL_fan+SPL_error_up) / 10) + 10 ** (SPL_othersources / 10))
        SPL_error_up=SPL_up-SPL

        f_error_dwn = interpolate.interp1d(df_sound_fan['PWM'], df_sound_fan['SPL_error_dwn'], fill_value='extrapolate')
        SPL_error_dwn = f_error_dwn(PWM)
        SPL_dwn = 10 * math.log10(2 * N * 10 ** ((SPL_fan + SPL_error_dwn) / 10) + 10 ** (SPL_othersources / 10))
        SPL_error_dwn = SPL_dwn - SPL



    return PWM,Vdot,Psat,SPL,SPL_error_up,SPL_error_dwn


database_path='fan_database/'
df=import_fan_data(database_path,'sanyo denki_test''.xlsx')
rho=1.2

df['C_F']=(df['Flowrate - [m^3/h]']/3600)/((df['Rotational Speed - [rpm]']/60)*2*math.pi*((df['Blade Diameter - [mm]']/1000)**3))

df['C_P']=((df['Flowrate - [m^3/h]']/3600)*df['Static Pressure - [Pa]'])/(rho*(((df['Rotational Speed - [rpm]']/60)*2*math.pi)**3)*((df['Blade Diameter - [mm]']/1000)**5))

df['C_H']=df['Static Pressure - [Pa]']/( (((df['Rotational Speed - [rpm]']/60)*2*math.pi)**2) *  ((df['Blade Diameter - [mm]']/1000)**2))

df['Sound Pressure - [Pa]']=(10**(df['Sound Level - [db-A]']/20))*20*10**(-6)

df['C_P - Sound']=((df['Flowrate - [m^3/h]']/3600)*df['Sound Pressure - [Pa]'])/(rho*(((df['Rotational Speed - [rpm]']/60)*2*math.pi)**3)*((df['Blade Diameter - [mm]']/1000)**5))

df['C_H - Sound']=df['Sound Pressure - [Pa]']/( (((df['Rotational Speed - [rpm]']/60)*2*math.pi)**2) *  ((df['Blade Diameter - [mm]']/1000)**2))


Vdot_inst=[
    356.6244,
    532.7254,
    721.7369,
    878.1878,
    1036.2305,
    1115.3166,
]

Psat_inst=[
    16.88857143,
    53.50571429,
    94.56714286,
    139.5128571,
    192.4642857,
    222.0757143,
]

list_of_fans={
    '9GV0824P1G03': 8
}


database=import_database(database_path)

ref=list(list_of_fans.keys())[0]
N=list_of_fans[ref]

Vdot_target=600
SPL_othersources=53

PWM,Vdot,Psat,SPL,SPL_error_up,SPL_error_dwn=PWM_predictor(ref,N,database,Vdot_target,Vdot_inst,Psat_inst,SPL_othersources)

df_fan=df[df['Model']==ref]

PWM_min,PWM_max=find_bounds(PWM,df_fan['PWM - [%]'].unique())
df_fan_min=df_fan[df_fan['PWM - [%]']==PWM_min]
df_fan_max=df_fan[df_fan['PWM - [%]']==PWM_max]

f_sound_dwn=interpolate.interp1d(df_fan_min['Flowrate - [m^3/h]'],df_fan_min['Sound Level - [db-A]'],fill_value='extrapolate')
f_sound_up=interpolate.interp1d(df_fan_max['Flowrate - [m^3/h]'],df_fan_max['Sound Level - [db-A]'],fill_value='extrapolate')

f_sound=interpolate.interp1d(np.array([PWM_min,PWM_max]),np.array([f_sound_dwn(Vdot_target/N),f_sound_up(Vdot_target/N)]),fill_value='extrapolate')
SPL_fan=f_sound(PWM)

SPL=10 * math.log10(2 * N * 10 ** ((SPL_fan) / 10) + 10 ** (SPL_othersources / 10))