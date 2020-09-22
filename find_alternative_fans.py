from database_creator import *

#region Fan
database_path = 'fan_database/'

database=import_database(database_path)

df_database=database['database']
df_flow=database['flow']
df_rpm=database['rpm']
df_sound=database['sound']

df_database=df_database[df_database['Supplier']=='Delta Eletronics']



v,f_min,f_mean,f_max=normalized_fans(database,plot='off')

Vdot_inst=[
    0,
    356.6244,
    532.7254,
    721.7369,
    878.1878,
    1036.2305,
    1115.3166,
]
Psat_inst=[
    0,
    16.88857143,
    53.50571429,
    94.56714286,
    139.5128571,
    192.4642857,
    222.0757143,
]



f_inst=interpolate.CubicSpline(Vdot_inst, Psat_inst, axis=0, bc_type='not-a-knot', extrapolate=None)

fan_tray_length = 548
fan_tray_width = 338
fan_flowrate_criterion=1000




df_sol = pd.DataFrame(
    columns=["Model", "Vdot_min","Psat_min","Vdot_mean","Psat_mean","Vdot_max","Psat_max","N"])

x=np.zeros(len(df_database))
y=np.zeros(len(df_database))


for i,ref in enumerate(df_database['Model'].unique()):
    df_fan = df_database[df_database['Model'] == ref]
    fan_length=df_fan['Length - [mm]'].values[0]
    fan_width=df_fan['Width - [mm]'].values[0]
    N_max_i=calc_max_n_fans(fan_tray_length,fan_tray_width,fan_length,fan_width)
    N=0
    Vdot_min=0; Vdot_mean=0 ; Vdot_max=0

    while (Vdot_min<fan_flowrate_criterion) and (N<N_max_i):
        N += 1

        Vdot_max = float(df_fan['Flowrate - [m^3/h]'])
        Psat_max = float(df_fan['Static Pressure - [Pa]'])

        performance_curves={'Min':[f_min,0,0,0],
                            'Mean':[f_mean,0,0,0],
                            'Max':[f_max,0,0,0]}
        for key in performance_curves:
            f=performance_curves[key][0]
            df=generate_performance_curve(f,Vdot_max,Psat_max)
            df['Vdot']=df['Vdot']*N
            performance_curves[key][1]=df

            Vdot_bigger = np.arange(min(df['Vdot']) * N, max(df['Vdot']) * N, (max(df['Vdot']) * N - min(df['Vdot']) * N) / 1000)
            f_fan = interpolate.interp1d(df['Vdot'], df['Psat'], axis=0, fill_value='extrapolate')

            E_abs=abs(f_inst(Vdot_bigger) - f_fan(Vdot_bigger))
            Vdot_wp = Vdot_bigger[np.argmin(E_abs)]
            Psat_wp = f_fan(Vdot_bigger)[np.argmin(E_abs)]
            performance_curves[key][2] = Vdot_wp
            performance_curves[key][3] = Psat_wp


        temp_sol = pd.DataFrame(
            columns=["Model","N", "Vdot_min","Psat_min","Vdot_mean","Psat_mean","Vdot_max","Psat_max","SPL_set"])

        SPL_set=10*math.log10(2*N*10**(df_fan['Sound Level - [db-A]']/10))

        temp_sol=temp_sol.append({'Model':ref,
                                  'N': N,
                                  'Vdot_min':performance_curves['Min'][2],
                                  'Psat_min':performance_curves['Min'][3],
                                  'Vdot_max':performance_curves['Max'][2],
                                  'Psat_max':performance_curves['Max'][3],
                                  'Vdot_mean':performance_curves['Mean'][2],
                                  'Psat_mean':performance_curves['Mean'][3],
                                  'SPL_set':SPL_set,

        },ignore_index=True)
        Vdot_min=float(temp_sol['Vdot_min'])
        Vdot_mean = float(temp_sol['Vdot_mean'])
        Vdot_max = float(temp_sol['Vdot_max'])

    print(ref+' : '+str(N)+'; Vdot_min = '+str(Vdot_min))
    df_sol=df_sol.append(temp_sol,sort=False)

    
    

df_sol=pd.merge(df_sol, df_database)

#Criteria for selecting optimal solution:

#df_sol=df_sol[df_sol['Vdot_mean']>=fan_flowrate_criterion]
#df_sol=df_sol[df_sol['PWM control function - [-]']=='Yes']
#df_sol=df_sol[df_sol['SPL_set']<=75]
#df_sol=df_sol.sort_values('SPL_set')
#df_sol=df_sol[df_sol['N']<=1]
#df_sol=df_sol[df_sol['Nominal Voltage - [V]']==24]
#df_sol=df_sol[df_sol['Depth - [mm]']<=38]

typologies_to_remove=['Centrifugal Fan','Splash Proof Centrifugal Fan','Blower']
for typology in typologies_to_remove:
    df_sol = df_sol[df_sol['Typology'] != typology]

#df_sol.to_csv(r'C:\Users\JoãoGonçaloCouto\selection.csv', index=False)


#ref=df_sol['Model'].unique()[0]

inspect_solution('EFB1324SHE-EP',database,df_sol,f_inst,f_min,f_mean,f_max,plot='On')