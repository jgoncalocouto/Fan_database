import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

global out


def colebrook_calc(epsilon, D_h, Re):
    f_0 = 0.02
    E_rel = 1
    stoppage_criteria = 1e-6
    while E_rel > stoppage_criteria:
        lambda_i = -2 * math.log((epsilon / (3.7 * D_h)) + (2.51 / (Re * f_0)))
        f = (1 / lambda_i) ** 2
        E_rel = abs(f - f_0) / f
        f_0 = f
    return f


def resistivity(material_name, T):
    # Import viscosities database
    database_path = 'fluid_properties/'
    elec_xls = pd.ExcelFile(database_path + 'eletric_properties.xlsx')
    data_elec = elec_xls.parse('resistivity'.lower()).values

    table_resistivity = {}

    table_resistivity['material'] = data_elec[1:, 0];
    table_resistivity['resistivity'] = data_elec[1:, 1];
    table_resistivity['alpha_resistivity'] = data_elec[1:, 2]

    m_index = list(table_resistivity['material']).index(material_name)
    rho_0 = table_resistivity['resistivity'][m_index]
    alpha = table_resistivity['alpha_resistivity'][m_index]

    rho = rho_0 * (1 + alpha * (T - 20))

    return rho


def thermal_properties_water(property_name, x_v, T):
    database_path = 'fluid_properties/'
    thermal_xls = pd.ExcelFile(database_path + 'thermal_properties_saturated_water.xlsx')
    df_thermal = thermal_xls.parse('h2o_saturated')

    exception_list = ['p_sat', 'h_fg', 'sigma', 'beta']

    # interpolate to get property

    if property_name in exception_list:
        out = np.interp(T, df_thermal['T_celsius'], df_thermal[property_name])
    else:
        out_f = np.interp(T, df_thermal['T_celsius'], df_thermal[property_name + '_f'])
        out_g = np.interp(T, df_thermal['T_celsius'], df_thermal[property_name + '_g'])
        out = out_f + ((out_g - out_f) / (1)) * (x_v)

    if T > max(df_thermal['T_celsius']) or T < min(df_thermal['T_celsius']):
        print('Warning: Temperature out of bounds!')
        print('Range of temperatures: [', str(min(df_thermal['T_celsius'])), ' ; ', str(max(df_thermal['T_celsius'])),
              ' ]')
        print('Value presented is saturated.')

    return out


def thermal_properties(property_name, fluid_name, T):
    # Import thermal properties database
    database_path = 'fluid_properties/'
    thermal_xls = pd.ExcelFile(database_path + 'thermal_properties.xlsx')
    df_thermal = thermal_xls.parse(fluid_name.lower())

    out = np.interp(T, df_thermal['T_celsius'], df_thermal[property_name])

    if (T > max(df_thermal['T_celsius']) or T < min(df_thermal['T_celsius'])):
        print('Warning: Temperature out of bounds!')
        print('Range of temperatures: [', str(min(df_thermal['T_celsius'])), ' ; ', str(max(df_thermal['T_celsius'])),
              ' ]')
        print('Value presented is saturated.')

    return out


def check_thermal_properties(fluid_name):
    # Import thermal properties database
    database_path = 'fluid_properties/'
    thermal_xls = pd.ExcelFile(database_path + 'thermal_properties.xlsx')
    df_thermal = thermal_xls.parse(fluid_name.lower())
    df_key = thermal_xls.parse('data_properties')

    n = len(df_thermal.columns) - 2
    if n / 2 > np.trunc(n / 2):
        n_sp = np.trunc(n / 2) + 1
        impar_window = 1
    else:
        n_sp = n / 2
        impar_window = 0

    fig = plt.figure()
    fig.suptitle("Thermal Properties for: " + fluid_name, fontweight="bold")
    l_axes = []
    for i in range(len(df_thermal.columns) - 2):
        if impar_window == 1 and i + 1 == len(df_thermal.columns) - 2:
            ax = fig.add_subplot(n_sp, 1, n_sp)
        else:
            ax = fig.add_subplot(n_sp, 2, i + 1)
        l_axes.append(ax)
        title = df_key.designation[i + 2] + ' - ' + df_key.symbol[i + 2] + ' - ' + df_key.units[i + 2]
        ax.set_title(title)
        ax.plot(df_thermal[df_key.property_name[0]], df_thermal[df_key.property_name[i + 2]])

    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    return df_thermal


def nusselt_internal_fd(ht):
    '''Inputs'''
    # Re , f , Pr , correlation_type

    if ht['Re'] >= 2300:
        ht['Nu'] = ((ht['f'] / 8) * (ht['Re'] - 1000) * ht['Pr']) / (
                    1 + (12.7 * (ht['f'] / 8) ** 0.5) * ((ht['Pr'] ** (2 / 3)) - 1))
        if ht['Pr'] < 0.5 or ht['Pr'] > 2000 or ht['Re'] < 3000 or ht['Re'] > 5 * 10 ** 6:
            print('Warning: Correlation used out of range in either Pr (Range = 0.5;2000) or Re (Range = 3000; 5e6)')
    elif (ht['Re'] < 2300 and ht.correlation_type == 'constant_temperature'):
        ht['Nu'] = 3.66
    elif (ht['Re'] < 2300 and ht.correlation_type == 'constant_flux'):
        ht['Nu'] = 4.36
    else:
        ht['Nu'] = 4.36
        print('No correlation type specified {contant_temperature , constant_flux}.')
        print('Value for nusselt number calculated for constant flux condition')

    return ht

def graetz_number(L_i,ht):
    L_characteristic=ht['L_characteristic'] ; Re=ht['Re'] ; Pr=ht['Pr']
    Grz=(L_characteristic/L_i)*Re*Pr
    ht['Grz']=Grz
    return ht

def grashoff_number(ht,T1,T2):
    '''Inputs'''
    # beta, L_characteristic, vu, T1, T2

    g=9.80665;
    beta=ht['beta']; L_characteristic = ht['L_characteristic']; vu = ht['vu'];
    Gr=(g*beta*abs(T1-T2)*L_characteristic**3)/(vu**2)
    ht['Gr']=Gr
    return ht

def richardson_number(ht):
    '''Inputs'''
    # Re , Gr
    
    ht['Ri']=ht['Gr']/(ht['Re']**2)
    return ht

def nusselt_internal_entry(ht):
    '''Inputs'''
    # Gz , Pr , Re
    Gz=ht['Gz'] ; Pr=ht['Pr'] ; Re=ht['Re']

    if Re>2300:
        print('Entry length effect on internal convection is not relevant in turbulent regimes. Correlation non-existent')
        print('Function for fully-developed flow will be launched')
        ht=nusselt_internal_fd(ht)
        return ht
    else:
        if Pr<0.1:
            print('Caution, Pr<0.1 -> Correlation not appropriate!')
        Nu=((3.66/(math.tanh(2.264*Gz**(-1/3)+1.7*Gz**(-2/3))))+0.0499*Gz*math.tanh(Gz**(-1)))/(math.tanh(2.432*(Pr**(1/6))*Gz**(-1/6)))
        ht['Nu']=Nu

    return ht

def nusselt_correction_fluids(ht):
    '''Inputs'''
    # T_m , T_s , mu_m , mu_s , Nu
    ht['Nu']=ht['Nu']*((ht['mu']/ht['mu_s'])**0.14)
    return ht

def thermal_properties_metals(property_name, solid_name):
    # Import thermal properties database
    database_path = 'fluid_properties/'
    thermal_xls = pd.ExcelFile(database_path + 'material_properties.xlsx')
    df_thermal = thermal_xls.parse('simplified')

    out = df_thermal[df_thermal['material'] == solid_name][property_name]

    return out

def calculate_Ts(Ti,To,DeltaT_lm):
    Ts=((-To*math.e**(To/DeltaT_lm))+(Ti*math.e**(Ti/DeltaT_lm)))/((-math.e**(To/DeltaT_lm))+(math.e**(Ti/DeltaT_lm)))
    return Ts

def nusselt_external_free(ht):
    Gr = ht['Gr'];
    Pr = ht['Pr'];
    Ra=Gr*Pr;
    Nu=(0.825+((0.387*Ra**(1/6))/((1+((0.492/Pr)**(9/16)))**(8/27))))**2
    ht['Nu']=Nu
    ht['Ra']=Ra
    return ht