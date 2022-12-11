
FDTD SIMULATION OF A LOSSLESS TRANSMISSION LINE - PROJECT EM I

This directory contains following subfolders/files:
	figures         : Storage folder with relevant plots of the simulation 
	animation       : Animation of the simulation

This simulation is performed with following parameters

Symbol      Value  Description
--------  -------  ------------------------------------------
d           0.1    Length of the transmission line [m]
v           2e+08  Velocity of the bit [m/s]
Rc          0      Characteristic impedance of the line [Ohm]
Rg        100      Generator resistance [Ohm]
Rl         25      Load resistanve [Ohm]
Cl          0      Load capacitance [F]
A           1      Bit amplitude [V]
Tbit        2e-10  But duration time [s]
tr          4e-11  Bit rising time [s]
D           1e-10  Delay of the bit [s]
N          50      Number of position steps
dt          1e-11  Time interval length [s]
M         400      Number of time steps
z_s         0.05   Position of sensor [frame]]
t_s        50      Time of snapshot [frame]