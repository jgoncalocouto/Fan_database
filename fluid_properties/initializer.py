from fluid_properties import *
import matplotlib.pyplot as plt
import numpy as np
import math
import pyromat as pm
from pyXSteam.XSteam import XSteam
from CoolProp.HumidAirProp import HAPropsSI

## Configure Pyromat
pm.config['unit_pressure']='Pa'
pm.config['unit_temperature']='C'
# Live example
working_fluid=pm.get('ig.air')
cp=working_fluid.cp(25,101325)



steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS) # m/kg/sec/°C/bar/W
'''
cp=steamTable.Cp_pt(25,1)
'''




