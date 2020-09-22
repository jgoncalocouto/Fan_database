from database_creator import *

#region Fan
database_path = 'fan_database/'
database=import_database(database_path)

df_database=database['database']
df_flow=database['flow']
df_rpm=database['rpm']
df_sound=database['sound']

#Vdot_min=(500*700/1000)*4*1.2
#Vdot_max=(600*700/1000)*4*1.2
Vdot_min=500
Vdot_max=600


SPL_othersources=53

#note: installation and fanset for grille n=2
list_of_fans={
    'EFB1524EG-00ERV': 4,
    'AFB1224GHEYNU': 8,
    'QFR1224GHE-SP01': 6,
}

list_of_prices={
    'EFB1524EG-00ERV': 30,
    'AFB1224GHEYNU': 22.05,
    'QFR1224GHE-SP01': 23.27000,
}

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

df_sol = pd.DataFrame(
    columns=["Model", "N","PWM","Vdot","Psat","SPL","Working Point","SPL_error_up","SPL_error_dwn","Price"], index=range(2*len(list_of_fans)))

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


for i,ref in enumerate(list_of_fans):
    N=list_of_fans[ref]
    price = list_of_prices[ref]

    try:
        PWM,Vdot,Psat,SPL,SPL_error_up,SPL_error_dwn=PWM_predictor(ref, N, database, Vdot_min, Vdot_inst, Psat_inst,SPL_othersources)
        df_sol['Model'][i]=ref
        df_sol['N'][i]=N
        df_sol["PWM"][i]=PWM
        df_sol["Vdot"][i]=Vdot
        df_sol["Psat"][i]=Psat
        df_sol["SPL"][i]=SPL
        df_sol["SPL_error_up"][i]=SPL_error_up
        df_sol["SPL_error_dwn"][i]=SPL_error_dwn
        df_sol["Working Point"][i]="Minimum"
        df_sol["Price"][i]=N*price

        f=i+len(list_of_fans)
        PWM,Vdot,Psat,SPL,SPL_error_up,SPL_error_dwn=PWM_predictor(ref, list_of_fans[ref], database, Vdot_max, Vdot_inst, Psat_inst,SPL_othersources)
        df_sol['Model'][f]=ref
        df_sol['N'][f]=N
        df_sol["PWM"][f]=PWM
        df_sol["Vdot"][f]=Vdot
        df_sol["Psat"][f]=Psat
        df_sol["SPL"][f]=SPL
        df_sol["SPL_error_up"][f]=SPL_error_up
        df_sol["SPL_error_dwn"][f]=SPL_error_dwn
        df_sol["Working Point"][f]="Maximum"
        df_sol["Price"][f] = N * price
    except:
        print(ref)


fig=plt.figure(figsize=(12,8))
ax1=plt.subplot(2,1,1)

#sns.scatterplot(x='Model',y='SPL',style='Working Point',hue='N',data=df_sol,ax=ax1)
#plt.errorbar(x=df_sol['Model'],y=df_sol['SPL'],yerr=)
plt.xticks(rotation=45, ha="right")
n=df_sol[df_sol['Working Point']=='Minimum']['Model']

ax1.fill_between(n,np.ones(len(n))*57.1, np.ones(len(n))*59,color='grey',alpha=0.2,label='Current Operating range')
ax1.fill_between(n,np.ones(len(n))*50,np.ones(len(n))*57.1,color='green',alpha=0.4,label='Desired Operating Range')
ax1.fill_between(n,np.ones(len(n))*59,np.ones(len(n))*70,color='indianred',alpha=0.4,label='Undesired Operating Range')


x=df_sol[df_sol['Working Point']=='Minimum']['Model']
y=df_sol[df_sol['Working Point']=='Minimum']['SPL']

err_up=df_sol[df_sol['Working Point']=='Maximum']['SPL_error_up'].values
err_dwn=df_sol[df_sol['Working Point']=='Maximum']['SPL_error_dwn'].values
yerr=np.vstack((err_up,err_dwn)).transpose()

ax1.plot(x,y,label='Minimum Working Point',color='teal',linestyle='-',marker='v')
ax1.plot(x,y+err_up,linestyle='dashed',color='teal')
ax1.plot(x,y-err_up,linestyle='dotted',color='teal')

x=df_sol[df_sol['Working Point']=='Maximum']['Model']
y=df_sol[df_sol['Working Point']=='Maximum']['SPL']

err_up=df_sol[df_sol['Working Point']=='Minimum']['SPL_error_up'].values
err_dwn=df_sol[df_sol['Working Point']=='Minimum']['SPL_error_dwn'].values
yerr=np.vstack((err_up,err_dwn)).transpose()

ax1.plot(x,y,label='Maximum Working Point',color='rebeccapurple',linestyle='-',marker='^')
ax1.plot(x,y+err_up,linestyle='dashed',color='rebeccapurple')
ax1.plot(x,y-err_up,linestyle='dotted',color='rebeccapurple')


ax1.set_ylabel('System Sound Level - [dB-A]')
ax1.set_title('Sound Level Ranges of potential alternatives vs Sound Level of Kiosk 50 kW)')
ax1.legend()

ax2=plt.subplot(2,1,2)
plt.xticks(rotation=45, ha="right")
x=df_sol[df_sol['Working Point']=='Maximum']['Model']
y=df_sol[df_sol['Working Point']=='Maximum']['Price']
n=df_sol[df_sol['Working Point']=='Maximum']['N']
ax2.scatter(x,y)
ax2.set_ylabel('Cost of fan tray - [$]')
ax2.set_title("Cost of solution")
for i,value in enumerate(n):
    ax2.annotate('N=%d , %.1f $ '%(n.iloc[i], y.iloc[i]),(x.iloc[i], y.iloc[i]))


df_sol.to_clipboard()
