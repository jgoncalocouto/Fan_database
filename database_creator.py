#region Import Modules

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
from scipy import interpolate
import math
import matplotlib.pyplot as plt
import statistics as stt
import scipy
from scipy import stats
import seaborn as sns
import os

#endregion

#region Functions:

database_path='fan_database/'


def import_fan_data(database_path,database_filename):

    data_xls = pd.ExcelFile(database_path + database_filename)
    list_of_fans = data_xls.sheet_names

    for i, fan in enumerate(list_of_fans):
        if i == 0:
            df_fan_general = data_xls.parse(fan)
        else:
            df_to_append = data_xls.parse(fan)
            df_fan_general = df_fan_general.append(df_to_append, ignore_index=True, sort=False)
            df_fan_general.dropna()
    return df_fan_general


def import_database(database_path):

    performance_aspects={
        'database': ['full',None],
        'flow':['flow',None],
        'sound':['sound',None],
        'rpm':['rpm',None],
    }


    list_of_files=os.listdir(database_path)
    supplier_list=[]
    for element in list_of_files:
        for key in performance_aspects:
            aspect=performance_aspects[key][0]

            if aspect in element:
                words=element.split('_')
                supplier_list.append(words[0])
    supplier_list=list(set(supplier_list))


    for key in performance_aspects:
        df_list=[]
        for i,supplier in enumerate(supplier_list):
            keyword=performance_aspects[key][0]
            try:
                df=import_fan_data(database_path,supplier+'_'+keyword+'.xlsx')
            except:
                df=None
            df_list.append(df)
        performance_aspects[key][1]=pd.concat(df_list,sort=False,ignore_index=True)

    dict_out = {}
    keys = performance_aspects.keys()
    for key in keys:
        dict_out[key] = performance_aspects[key][1]
    return dict_out


def find_bounds(value,list_of_values):
    nextHighest = lambda seq, x: min([(i - x, i) for i in seq if x <= i] or [(0, None)])[1]
    nextLowest = lambda seq, x: min([(x - i, i) for i in seq if x >= i] or [(0, None)])[1]

    lower_bound=nextLowest(list_of_values,value)
    upper_bound=nextHighest(list_of_values,value)

    return lower_bound, upper_bound


def VdotPsat_from_rpm(N,N_lower,N_upper,Vdot_lower,Psat_lower,Vdot_upper,Psat_upper):

    Psat_1=(Psat_lower*(N**2))/(N_lower**2)
    Vdot_1=(Vdot_lower*(N))/(N_lower)
    f_curve1=interpolate.interp1d(Vdot_1,Psat_1,fill_value='extrapolate') # curve for RPM=N built from bellow

    Psat_2=(Psat_upper*(N**2))/(N_upper**2)
    Vdot_2=(Vdot_upper*(N))/(N_upper)
    f_curve2=interpolate.interp1d(Vdot_2,Psat_2,fill_value='extrapolate') # curve for RPM=N built from above

    Vdot_min=max(min(Vdot_1),min(Vdot_2))
    Vdot_max=min(max(Vdot_1),max(Vdot_2))
    Vdot_step=(Vdot_max-Vdot_min)/max(len(Vdot_1),len(Vdot_2))

    Vdot=np.arange(Vdot_min,Vdot_max,Vdot_step)
    Psat=(f_curve1(Vdot)+f_curve2(Vdot))/2 # curve for RPM=N built from the average between above and bellow

    f_aggregate=interpolate.interp1d(Vdot,Psat,fill_value='extrapolate')
    finv_aggregate=interpolate.interp1d(Psat,Vdot,fill_value='extrapolate')
    Vdot=np.append(0,Vdot)
    Psat=f_aggregate(Vdot)
    Psat = np.append(Psat,0)
    Vdot = finv_aggregate(Psat)

    data=np.concatenate((Vdot,Psat),axis=0).reshape(2,len(Vdot))
    data=data.transpose()
    df_out=pd.DataFrame(data=data,columns=["Vdot","Psat"])
    return df_out


def fan_characteristic(database,reference,control_parameter,control_value):
    df_flow=database['flow']
    df_rpm=database['rpm']
    df_full=database['database']
    df_fan_flow=df_flow[df_flow['Model']==reference.upper()]
    df_fan_rpm=df_rpm[df_rpm['Model']==reference.upper()]
    df_fan_full=df_full[df_full['Model']==reference.upper()]

    if df_fan_flow.empty:
        print('Fan reference not know in the Flowrate vs Static Pressure Database.')
        print('Function execution terminated.')
        return
    else:
        if control_parameter.upper() == 'PWM':
            try:
                f_RPM = interpolate.interp1d(df_fan_rpm['PWM - [%]'].unique(), df_fan_rpm['Rotational Speed - [rpm]'].unique())
                PWM = control_value
                N = f_RPM(PWM)
            except:
                if control_value==100:
                    df_sol=df_fan_flow[['Flowrate - [m^3/h]','Static Pressure - [Pa]']].copy(deep=False)
                    df_sol.rename(columns={
                        'Flowrate - [m^3/h]':'Vdot',
                        'Static Pressure - [Pa]':'Psat',
                    })
                    df_sol=df_sol.reset_index(drop=True)
                    return df_sol


        elif control_parameter.upper() == 'RPM':
            try:
                f_RPM = interpolate.interp1d(df_fan_rpm['PWM - [%]'].unique(),
                                         df_fan_rpm['Rotational Speed - [rpm]'].unique())
                f_PWM = interpolate.interp1d(df_fan_rpm['Rotational Speed - [rpm]'].unique(),
                                         df_fan_rpm['PWM - [%]'].unique())
                PWM=f_PWM(control_value)
                N=control_value
            except:
                if (df_fan_rpm.empty==True and np.isnan(df_fan_full['Rotational Speed - [rpm]'].values[0])==True):
                    print('PWM vs RPM curve not available. Please check database.')
                    print("Function execution terminated")
                    return
                elif (df_fan_rpm.empty==True and df_fan_full['Rotational Speed - [rpm]'].values[0]==control_value):
                    df_sol=df_fan_flow[df_fan_flow['PWM -[%]']==100]
                    df_sol=df_sol[['Flowrate - [m^3/h]','Static Pressure - [Pa]']].copy(deep=False)
                    df_sol.rename(columns={
                        'Flowrate - [m^3/h]':'Vdot',
                        'Static Pressure - [Pa]':'Psat',
                    })
                    df_sol=df_sol.reset_index()
                    df_sol=df_sol.drop(columns=['index'])
                    return df_sol
                else:
                    print('PWM vs RPM curve not available. Please check database.')
                    print("Function execution terminated")
                    return
        else:
            print("Control parameter not recognized.")
            print("Function execution terminated")
            return

        PWM_lower, PWM_upper = find_bounds(PWM, df_fan_flow['PWM - [%]'].unique())
        N_lower, N_upper = f_RPM(PWM_lower), f_RPM(PWM_upper)

        if PWM_lower == None:
            print(
                "Control Value is bellow the minimum specified by supplier. Data will be given for the minimum specified by supplier")
            PWM_lower = PWM_upper
        elif PWM_upper == None:
            print(
                "Control Value is above the maximum specified by supplier. Data will be given for the maximum specified by supplier")
            PWM_upper = PWM_lower

        lower_df = df_fan_flow[df_fan_flow['PWM - [%]'] == PWM_lower]
        upper_df = df_fan_flow[df_fan_flow['PWM - [%]'] == PWM_upper]

        df_sol = VdotPsat_from_rpm(N, N_lower, N_upper, lower_df['Flowrate - [m^3/h]'],
                                    lower_df['Static Pressure - [Pa]'], upper_df['Flowrate - [m^3/h]'],
                                    upper_df['Static Pressure - [Pa]'])
        return df_sol


def sound_level_predictor(N_min,N_max,SPL_min,SPL_max,N):
    N_vector = np.array([N_min, N_max])
    SPL_vector=np.array([SPL_min,SPL_max])

    m, b, r_value, p_value, std_err = stats.linregress(list(map(math.log,N_vector)), SPL_vector)
    SPL=math.log(N)*m+b

    return SPL


def fan_sound(database,reference):
    df_flow = database['flow']
    df_rpm = database['rpm']
    df_full = database['database']
    df_sound = database['sound']
    df_fan_flow = df_flow[df_flow['Model'] == reference.upper()]
    df_fan_rpm = df_rpm[df_rpm['Model'] == reference.upper()]
    df_fan_full = df_full[df_full['Model'] == reference.upper()]
    df_fan_sound = df_sound[df_sound['Model'] == reference.upper()]

    PWM_yerr_up=[
        0,
        10,
        20,
        25.3,
        34,
        40,
        58.6,
        80,
        88,
        100,
    ]
    SPL_yerr_up=[
        0,
        4.8,
        7.8,
        3.8,
        2.42,
        1.99,
        0.97,
        0.24,
        0.25,
        0,
    ]
    PWM_yerr_dwn=[
        0,
        7.25,
        14,
        34,
        46,
        71.83,
        90,
        92.75,
        100,
    ]
    SPL_yerr_dwn=[
        0,
        - 3.48,
        - 4.77,
        - 4.53,
        - 4.25,
        - 2.58,
        - 1.78,
        - 1.58,
        0,
    ]
    f_yerr_dwn=interpolate.interp1d(PWM_yerr_dwn,SPL_yerr_dwn,fill_value='extrapolate')
    f_yerr_up=interpolate.interp1d(PWM_yerr_up,SPL_yerr_up,fill_value='extrapolate')

    if df_fan_full.empty:
        print('Reference not found in database. Please check spelling.')
        return

    if df_fan_rpm.empty:
        if df_fan_sound.empty and np.isnan(df_fan_full['Sound Level - [db-A]'].values[0]):
            print('No sound level data available for this fan reference!')
            return
        elif df_fan_sound.empty and not np.isnan(df_fan_full['Sound Level - [db-A]'].values[0]):
            print('No RPM or PWM vs SPL curve available. Only nominal value will be presented')
            data = [100, df_fan_full['Rotational Speed - [rpm]'].values[0],
                    df_fan_full['Sound Level - [db-A]'].values[0]]
            df_sol = pd.DataFrame(columns=["PWM", "RPM", "SPL"], index=[1])
            df_sol.loc[1]=data
            df_sol["SPL_error_up"] = 0
            df_sol["SPL_error_dwn"] = 0
            return df_sol
    else:

        if not df_fan_sound.empty and len(df_fan_sound)>2:
            f_RPM = interpolate.interp1d(df_fan_rpm['PWM - [%]'].unique(),
                                         df_fan_rpm['Rotational Speed - [rpm]'].unique(), fill_value="extrapolate")
            df_sol=df_fan_sound.copy()
            df_sol=df_sol.drop('Model', axis=1)
            df_sol['RPM'] = f_RPM(df_sol['PWM - [%]'])
            df_sol = df_sol.rename(columns={'PWM - [%]': 'PWM', 'Sound Level - [db-A]': 'SPL'})
            df_sol=df_sol.reset_index(drop=True)
            df_sol["SPL_error_up"]=0
            df_sol["SPL_error_dwn"] =0
            return df_sol
        elif len(df_fan_sound) == 2:
            f_RPM = interpolate.interp1d(df_fan_rpm['PWM - [%]'].unique(),
                                         df_fan_rpm['Rotational Speed - [rpm]'].unique(), fill_value="extrapolate")
            print('Sound level data not available, linear regression beased on LN(RPM) vs SPL to be used')
            N_min, N_max = min(df_fan_rpm['Rotational Speed - [rpm]'].unique()), max(
                df_fan_rpm['Rotational Speed - [rpm]'].unique())
            SPL_min, SPL_max = min(df_fan_sound['Sound Level - [db-A]'].unique()), max(
                df_fan_sound['Sound Level - [db-A]'].unique())
            PWM_min, PWM_max = min(df_fan_rpm['PWM - [%]'].unique()), max(df_fan_rpm['PWM - [%]'].unique())
            PWM_vector = np.arange(PWM_min, PWM_max + 1, 1)
            N_vector = f_RPM(PWM_vector)
            SPL_vector = np.array([])
            for N in N_vector:
                SPL_vector = np.append(SPL_vector,
                                       sound_level_predictor(N_min, N_max, SPL_min, SPL_max, N)
                                       )
            data = np.concatenate((PWM_vector, N_vector, SPL_vector), axis=0).reshape(3, len(N_vector))
            data = data.transpose()
            df_sol = pd.DataFrame(data=data, columns=["PWM", "RPM", "SPL"])
            df_sol["SPL_error_up"] = f_yerr_up(df_sol["PWM"])
            df_sol["SPL_error_dwn"] = f_yerr_dwn(df_sol["PWM"])
            return df_sol
        else:
            print('No RPM or PWM vs SPL curve available. Only nominal value will be presented')
            data = [df_fan_rpm['PWM - [%]'].values[0], df_fan_rpm['Rotational Speed - [rpm]'].values[0],
                    df_fan_full['Sound Level - [db-A]'].values[0]]
            df_sol = pd.DataFrame(columns=["PWM", "RPM", "SPL"], index=[1])
            df_sol.loc[1]=data
            df_sol["SPL_error_up"] = 0
            df_sol["SPL_error_dwn"] = 0
            return df_sol


def fan_details(database,reference):
    df_full = database['database']
    df_fan_full = df_full[df_full['Model'] == reference.upper()]

    details={
        'Supplier':df_fan_full['Supplier'].unique()[0],
        'Typology': df_fan_full['Typology'].unique()[0],
        'Reference': df_fan_full['Model'].unique()[0],
        'Voltage': df_fan_full['Nominal Voltage - [V]'].unique()[0],
        'Dimensions': (df_fan_full['Length - [mm]'].unique()[0],df_fan_full['Width - [mm]'].unique()[0],df_fan_full['Depth - [mm]'].unique()[0])
    }

    return details


def fan_summary_plot(database,reference):
    df_flow = database['flow']

    df_fan_flow = df_flow[df_flow['Model'] == reference.upper()]

    df_spl=fan_sound(database,reference)
    df_spl=df_spl.round({'PWM': 1, 'SPL': 1, 'RPM': 0})


    try:
        details = fan_details(database, reference)
        fig = plt.figure(figsize=(12, 8))
        ax_1 = fig.add_subplot(1, 2, 1)

        if df_fan_flow.empty:
            print("No performance data available for plotting")
            return

        sns.lineplot(x='Flowrate - [m^3/h]', y='Static Pressure - [Pa]', style='PWM - [%]', data=df_fan_flow, axes=ax_1)
        ax_2 = fig.add_subplot(2, 2, 2)

        sns.lineplot(x='PWM', y='SPL', data=df_spl, axes=ax_2, color='indianred')
        ax_2.set_xlabel('PWM - [%]')
        ax_2.set_ylabel('Sound Level - [dB-A]')
        fig.suptitle(
            details['Supplier'] + ' | ' + details['Reference'] + ' | Dimensions:' + str(
                int(details['Dimensions'][0])) + ' X ' + str(int(details['Dimensions'][1])) + ' X ' + str(
                int(details['Dimensions'][2])) + ' [mm]',
            fontsize=16
        )

        ax_3 = fig.add_subplot(2, 2, 4)
        if len(df_spl) > 10:
            to_keep = np.arange(min(df_spl['PWM']), max(df_spl['PWM']) + 10, 10)
            f_rpm = interpolate.interp1d(df_spl['PWM'], df_spl['RPM'], fill_value='extrapolate')
            f_spl = interpolate.interp1d(df_spl['PWM'], df_spl['SPL'], fill_value='extrapolate')
            df_spl = pd.DataFrame(columns=['PWM', 'RPM', 'SPL'],
                                  data=np.array([to_keep, f_rpm(to_keep), f_spl(to_keep)]).transpose())

        ax_3.table(cellText=df_spl.values, colLabels=df_spl.columns, loc='center', cellLoc='center',
                   colColours=plt.cm.RdPu(np.linspace(0, 0.5, 4)))
        ax_3.axis("off")



        ax = [ax_1, ax_2, ax_3]
        return fig, ax
    except:
        print('Performance data not available to plot')


def statistics_plot(dtf,x,suptitle=None):

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False,figsize=(9,5))
    if suptitle==None:
        suptitle=x
    fig.suptitle(suptitle, fontsize=20)
    ### distribution
    ax[0].set_title('distribution')
    variable = dtf[x]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
    des = dtf[x].describe()
    ax[0].axvline(des["25%"], ls='--')
    ax[0].axvline(des["mean"], ls='--')
    ax[0].axvline(des["75%"], ls='--')
    ax[0].grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: " + des["min"], "25%: " + des["25%"], "mean: " + des["mean"], "75%: " + des["75%"],
                     "max: " + des["max"]))
    ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right",
               bbox=dict(boxstyle='round', facecolor='white', alpha=1))  ### boxplot

    ax[1].title.set_text('Boxplot')
    dtf.boxplot(column=x, ax=ax[1])
    plt.show()

    return fig,ax


def pwm_curves_error(database):
    df_flow=database['flow']
    E_rel = []
    E_abs = []

    df_error = pd.DataFrame(columns=["Model", "Reference Flowrate - [m^3/h]", "Reference Static Pressure - [Pa]",
                                     'Normalized Flowrate', 'Absolute Deviation', 'Relative Deviation'])
    count = 0
    for reference in df_flow['Model'].unique():
        df_fan = df_flow[df_flow['Model'] == reference]
        if len(df_fan['PWM - [%]'].unique()) > 2:
            PWM_value = df_fan['PWM - [%]'].unique()[1]
            df_fan = df_fan[df_fan['PWM - [%]'] == PWM_value]
            Vdot_50 = df_fan['Flowrate - [m^3/h]']
            P_sat50 = df_fan['Static Pressure - [Pa]']
            df_flow_49 = fan_characteristic(database, reference, 'PWM', PWM_value - 0.0001)
            f_49 = interpolate.interp1d(df_flow_49['Psat'], df_flow_49['Vdot'], fill_value='extrapolate')
            df_flow_51 = fan_characteristic(database, reference, 'PWM', PWM_value + 0.0001)
            f_51 = interpolate.interp1d(df_flow_51['Psat'], df_flow_51['Vdot'], fill_value='extrapolate')
            Vdot_49 = f_49(P_sat50);
            Vdot_51 = f_51(P_sat50)
            E_rel_i = (abs(((Vdot_49 - Vdot_50) / P_sat50) * 100) + abs(((Vdot_51 - Vdot_50) / Vdot_50) * 100)) / 2
            E_abs_i = (abs(((Vdot_49 - Vdot_50) / 1)) + abs(((Vdot_51 - Vdot_50) / 1))) / 2
            E_rel.append(list(E_rel_i))
            E_abs.append(list(E_abs_i))

            df_i = pd.DataFrame({"Model": [reference] * len(Vdot_50),
                                 "Reference Flowrate - [m^3/h]": list(Vdot_50),
                                 "Reference Static Pressure - [Pa]": list(P_sat50),
                                 'Normalized Flowrate': list(Vdot_50 / max(Vdot_50)),
                                 'Absolute Deviation': list(E_abs_i),
                                 'Relative Deviation': list(E_rel_i)})

            df_error = df_error.append(df_i, ignore_index=True, sort=False)

    df_error['Relative Deviation'] = list(
        map(lambda x: np.nan if (x == np.inf or x >= 300) else x, df_error['Relative Deviation']))

    bins = {
        '25%': [0, 0.25],
        '25 - 50%': [0.25001, 0.5],
        '50 - 75%': [0.5001, 0.75],
        '75 - 100%': [0.75001, 1]}
    df_error["Class"] = ''

    for key in bins:
        low = bins[key][0]
        up = bins[key][1]
        df_error.loc[low <= df_error['Normalized Flowrate'], 'Class'] = key

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        'Relative Error on Flowrate caused by using Turbomachines relations to predict Flowrate vs Static Pressure at different PWM',
        fontsize=16)
    ax_1 = fig.add_subplot(1, 2, 1)
    g = sns.boxplot(x='Class', y='Relative Deviation', data=df_error, order=list(bins.keys()), ax=ax_1,
                    palette="Blues")
    ax_1.set_ylim([0, 150])
    ax_1.set_xlabel('Normalized Flowrate')
    ax_2 = fig.add_subplot(2, 2, 2)
    f = sns.scatterplot(x='Normalized Flowrate', y='Relative Deviation', hue='Class', data=df_error, ax=ax_2)
    ax_2.legend(loc='upper right')
    ax_3 = fig.add_subplot(2, 2, 4)
    for region in bins.keys():
        h = sns.distplot(df_error.loc[df_error['Class'] == region, 'Relative Deviation'], ax=ax_3)

    return df_error


def normalized_fans(database,plot='off'):
    df_flow=database['flow']
    df_flow=df_flow.set_index('Model')
    df_database=database['database']
    df_database=df_database.set_index('Model')

    df_all=pd.merge(df_database, df_flow, left_index=True, right_index=True)

    df_model = pd.DataFrame(
        columns=["Model", "Normalized Maximum Flowrate", "Normalized Maximum Static Pressure", 'Length - [mm]',
                 'Depth - [mm]','Sound Level - [db-A]'])
    for ref in df_all.index.unique():
        temp_df = pd.DataFrame(columns=["Model", "Normalized Maximum Flowrate", "Normalized Maximum Static Pressure"])
        df_fan = df_all.loc[ref]
        df_fan = df_fan[df_fan['PWM - [%]'] == max(df_fan['PWM - [%]'])]

        temp_df['Model'] = df_fan.index
        temp_df['Length - [mm]'] = df_fan["Length - [mm]"].values
        temp_df['Depth - [mm]'] = df_fan["Depth - [mm]"].values
        temp_df['Sound Level - [db-A]'] = df_fan["Sound Level - [db-A]"].values
        temp_df['Normalized Maximum Flowrate'] = df_fan["Flowrate - [m^3/h]_y"].values / max(df_fan["Flowrate - [m^3/h]_y"].values)
        temp_df['Normalized Maximum Static Pressure'] = df_fan["Static Pressure - [Pa]_y"].values / max(
            df_fan["Static Pressure - [Pa]_y"].values)

        df_model = df_model.append(temp_df)
    df_model=df_model.reset_index(drop=True)

    V_max = np.arange(0.1, 1 + 0.1, 0.1)
    V_min = np.arange(0, 1, 0.1)
    Vdot_min = [0]
    Psat_min = [1]
    Vdot_max = [0]
    Psat_max = [1]

    for i, value in enumerate(V_min):
        df_i = df_model[
            (df_model['Normalized Maximum Flowrate'] > V_min[i]) & (df_model['Normalized Maximum Flowrate'] < V_max[i])]

        Psat_min.append(min(df_i['Normalized Maximum Static Pressure']))
        id_min = df_i['Normalized Maximum Static Pressure'].idxmin()
        Vdot_min.append(df_i['Normalized Maximum Flowrate'][id_min])

        Psat_max.append(max(df_i['Normalized Maximum Static Pressure']))
        id_max = df_i['Normalized Maximum Static Pressure'].idxmax()
        Vdot_max.append(df_i['Normalized Maximum Flowrate'][id_max])

    Vdot_min.append(1)
    Psat_min.append(0)
    Vdot_max.append(1)
    Psat_max.append(0)

    Vdot_min, Vdot_max, Psat_min, Psat_max = np.array(Vdot_min), np.array(Vdot_max), np.array(Psat_min), np.array(
        Psat_max)
    f_min = interpolate.CubicSpline(Vdot_min, Psat_min, axis=0, bc_type='not-a-knot', extrapolate=None)
    f_max = interpolate.CubicSpline(Vdot_max, Psat_max, axis=0, bc_type='not-a-knot', extrapolate=None)
    v = np.arange(0, 1 + 0.0001, 0.0001)
    f_mean=interpolate.CubicSpline(v, (f_min(v)+f_max(v))/2, axis=0, bc_type='not-a-knot', extrapolate=None)


    if plot == 'on':
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        g = sns.scatterplot(x='Normalized Maximum Flowrate', y='Normalized Maximum Static Pressure', hue='Sound Level - [db-A]',
                        data=df_model, ax=ax)
        plt.plot(v, f_min(v), color='black', label='Minimum Envelope')
        plt.plot(v, f_max(v), color='red', label='Maximum Envelope')
        plt.plot(v, f_mean(v), color='blue', label='Mean Envelope')
    return v,f_min,f_mean,f_max


def generate_performance_curve(f_model,Vdot_max,Psat_max):

    Start=0;Step=(Vdot_max-0)/1000;Stop=Vdot_max+Step
    Vdot=np.arange(start=Start,stop=Stop,step=Step)
    Psat=f_model(Vdot/Vdot_max)*Psat_max

    data=np.concatenate((Vdot,Psat),axis=0).reshape(2,len(Vdot))
    data=data.transpose()
    df_out=pd.DataFrame(data=data,columns=["Vdot","Psat"])
    return df_out


def calc_max_n_fans(length,width,length_fan,width_fan):
    n_max_row=length//length_fan
    n_max_column=width//width_fan
    N=n_max_row*n_max_column
    return N


def inspect_solution(ref,database,df_solution,f_inst,f_min,f_mean,f_max,plot='On',N=None):

    df_flow = database['flow']
    df_rpm = database['rpm']
    df_full = database['database']
    df_sound = database['sound']
    df_fan_flow = df_flow[df_flow['Model'] == ref.upper()]
    df_fan_rpm = df_rpm[df_rpm['Model'] == ref.upper()]
    df_fan_full = df_full[df_full['Model'] == ref.upper()]
    df_fan_sound = df_sound[df_sound['Model'] == ref.upper()]


    df_solution_details=df_solution[df_solution['Model']==ref]

    if N is None:
        N=df_solution_details['N'].values[0]

    Vdot_max = float(df_solution_details['Flowrate - [m^3/h]'])
    Psat_max = float(df_solution_details['Static Pressure - [Pa]'])

    performance_curves={'Max': [f_max,[]],
                        'Mean': [f_mean,[]],
                        'Min': [f_min,[]],
                       }

    if not df_fan_flow.empty:
        df_flow=df_fan_flow[df_fan_flow['PWM - [%]']==100]
    else:
        print('Performance data not available. Check reference!')
        return

    for key in performance_curves:
        df=generate_performance_curve(performance_curves[key][0], Vdot_max, Psat_max)
        df['Vdot']=df['Vdot']*N
        performance_curves[key][1]=df

    Vdot=df_flow['Flowrate - [m^3/h]']*N
    Psat=df_flow['Static Pressure - [Pa]']
    f_fan=interpolate.interp1d(Vdot, Psat, axis=0, fill_value='extrapolate')
    Vdot_bigger=np.arange(min(Vdot)*N,max(Vdot)*N,(max(Vdot)*N-min(Vdot)*N)/1000)
    E_abs = abs(f_inst(Vdot_bigger) - f_fan(Vdot_bigger))
    Psat_workingpoint=f_inst(Vdot_bigger)[np.argmin(E_abs)]
    Vdot_workingpoint=Vdot_bigger[np.argmin(E_abs)]


    if plot.lower()=='on':
        plt.figure()
        plt.plot(df_flow['Flowrate - [m^3/h]'] * N,
                 df_flow['Static Pressure - [Pa]'], color='pink',
                 label='Real Curve')
        plt.plot(performance_curves['Min'][1]['Vdot'], f_inst(performance_curves['Min'][1]['Vdot']), color='black', label='Installation')

        list_of_colors={
            'Max': 'red',
            'Mean':'green',
            'Min':'blue',
        }
        for key in performance_curves:
            df=performance_curves[key][1]
            plt.plot(df['Vdot'], df['Psat'], color=list_of_colors[key], label=key)

        x,y=Vdot_workingpoint,Psat_workingpoint
        plt.text(x, y, '('+"{:.1f}".format(x)+' , '+"{:.0f}".format(y)+')',backgroundcolor = 'lavenderblush')

        plt.xlabel('Flowrate - [m^3/h]')
        plt.ylabel('Static Pressure - [Pa]')
        plt.title(df_solution_details['Supplier'].values[0]+', Reference: ' + ref + ' N = ' + str(N))
        plt.legend()


def hydraulic_working_point(Vdot_inst,Psat_inst,Vdot_fan,Psat_fan,plot='off',N=1):
    f_inst = interpolate.CubicSpline(Vdot_inst, Psat_inst, axis=0, bc_type='not-a-knot', extrapolate=True)
    f_fan=interpolate.interp1d(Vdot_fan*N, Psat_fan, fill_value='extrapolate')

    start=min([min(Vdot_inst),min(Vdot_fan*N)])
    stop = max([max(Vdot_inst), max(Vdot_fan*N)])
    step=(stop-start)/10000
    Vdot=np.arange(start,stop,step)

    E_abs=abs(f_inst(Vdot)-f_fan(Vdot))

    working_point_Vdot=Vdot[np.nanargmin(E_abs)]
    working_point_Psat = f_fan(Vdot[np.nanargmin(E_abs)])

    if plot=='on':
        fig=plt.figure()
        ax=fig.add_subplot(111)
        line1=ax.plot(Vdot,f_inst(Vdot),label='Installation Curve',color='black')
        line2=ax.plot(Vdot,f_fan(Vdot),label='Fan Curve',color='red')
        line3=ax.scatter(working_point_Vdot,working_point_Psat,label='Working Point',marker='D',color='teal')
        ax.legend()
        ax.set_xlabel('Flowrate - [m^3/h]')
        ax.set_ylabel('Static Pressure - [Pa]')
        ax.set_title('Working Point Prediction: N_fans = '+str(N))
        ax.set_ylim([0,max(Psat_fan)*1.1])

    return working_point_Vdot,working_point_Psat






database=import_database(database_path)

reference='AFB1224EHE-EP'
df_sol=fan_characteristic(database,reference,'PWM',100)

