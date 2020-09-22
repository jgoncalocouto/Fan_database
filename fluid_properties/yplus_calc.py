from fluid_properties import *
import matplotlib.pyplot as plt
import numpy as np
import math
import pyromat as pm
from pyXSteam.XSteam import XSteam
from CoolProp.HumidAirProp import HAPropsSI

def wall_element_calc(case,y_plus,V_inf,L_characteristic,epsilon,fluid_name,T1,p1):
    # get working_fluid properties
    pm.config['unit_pressure'] = 'Pa'
    pm.config['unit_temperature'] = 'C'
    working_fluid = pm.get('ig.' + fluid_name)
    rho=working_fluid.d(T=T1, p=p1)
    mu=working_fluid.mu = thermophysical_properties('mu', fluid_name, T1)
    Re=(rho*V_inf*L_characteristic)/mu

    if case=="pipe_flow":
        cf=colebrook_calc(epsilon, L_characteristic, Re)/4
        tau_w=cf*0.5*rho*V_inf**2
        u_star=math.sqrt(tau_w/rho)
        y=(y_plus*mu)/(rho*u_star)
    return y
y=wall_element_calc('pipe_flow',30,30,0.001,0,'air',15,101325)
print(y)





