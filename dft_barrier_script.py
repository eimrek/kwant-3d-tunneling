import numpy as np
import time
import matplotlib
matplotlib.use('Agg') # without x environment
import matplotlib.pyplot as plt

import kwant

import os
import argparse
import sys
sys.path.append('./modules/')
import wraparound

import scipy as sp
import scipy.ndimage
import scipy.constants
# CONSTANTS #
kB = sp.constants.value("Boltzmann constant in eV/K") # unit: eV/K
qe = sp.constants.value("elementary charge") # unit: C
me = sp.constants.value("electron mass")/qe*1e-18 #unit: eV*s^2/nm^2
hP = sp.constants.value("Planck constant in eV s") #unit: eV*s
hbar = hP/(2*sp.pi) #unit: eV*s
eps0 = sp.constants.value("electric constant")*qe*1e-9 #unit: C^2/(eV*nm)


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parse arguments #
parser = argparse.ArgumentParser(
    description="Calculates transmission coefficient for "\
                "electrons through specified 3d potential barrier.")

parser.add_argument('filepath', metavar='f', help='Calculation input file')
args = parser.parse_args()

# Read all system data from file
# Also Kwant lattice spacing

potential_data_file = args.filepath

file_name = os.path.splitext(potential_data_file.split("/")[-1])[0]
result_file = "./result_data/"+file_name

potential_data = np.load(potential_data_file)
pot    = potential_data["pot"]
box    = potential_data["box"]
num    = potential_data["num"]
a      = potential_data["par"][0]
e_zero = potential_data["par"][1]

energies_global = potential_data["energies_global"]

kx_sl = 0.0
ky_sl = 0.0

# -----------------------------------------------------------------------------
# Energy range
e_num = len(energies_global)
start_i = int(np.round(e_num/size*rank))
end_i = int(np.round(e_num/size*(rank+1)))
energies = energies_global[start_i:end_i]


# -----------------------------------------------------------------------------
# Kwant setup
L = num

def onsite_potential(site):
    (x,y,z) = site.pos
    [ix, iy, iz] = np.rint(np.array([x,y,z])/a).astype(int)
    return pot[ix, iy, iz]

t = hbar**2/(2*me*a**2) # units: eV

lat = kwant.lattice.general([(a, 0, 0), (0, a, 0), (0, 0, a)])

sys = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((L[0], 0, 0)), lat.vec((0, L[1], 0))))
sys[(lat(i,j,k) for i in range(L[0]) for j in range(L[1]) for k in range(L[2]))] = lambda p: 6*t+onsite_potential(p)
sys[lat.neighbors(1)] = -t

sys = wraparound.wraparound(sys)

#### Define and attach leads. ####
lead = kwant.Builder(kwant.TranslationalSymmetry((0, 0, -a), lat.vec((L[0], 0, 0)), lat.vec((0, L[1], 0))))
lead[(lat(i, j, 0) for i in range(L[0]) for j in range(L[1]))] = 6 * t
lead[lat.neighbors()] = -t
lead = wraparound.wraparound(lead, keep=0)
sys.attach_lead(lead)
sys.attach_lead(lead.reversed())
# --------------------------------------------------------------- #

if rank == 0:
    print("Plotting kwant system...")
    kwant.plot(sys, site_color=onsite_potential, cmap='jet', site_size=0.4, fig_size=(15, 9), file=("./fig/"+file_name+".png"))

sys = sys.finalized()

# -----------------------------------------------------------------------------
# Starting calculation

transmission = []
submatrices = []
num_prop = []

for energy in energies:
    start_time = time.time()
    smatrix = kwant.smatrix(sys, energy, [kx_sl, ky_sl])
    submatrices.append(smatrix.submatrix(1, 0))
    transmission.append(smatrix.transmission(1, 0))
    num_prop.append(smatrix.num_propagating(0))
    print("processor: %d; energy: %2.2f; time: %3.2f"%(rank, energy, time.time()-start_time))

transmission = comm.gather(transmission, root=0)
submatrices = comm.gather(submatrices, root=0)
num_prop = comm.gather(num_prop, root=0)

if rank == 0:
    # Flatten python lists
    transmission = [item for sublist in transmission for item in sublist]
    submatrices = [item for sublist in submatrices for item in sublist]
    num_prop = [item for sublist in num_prop for item in sublist]

    transmission = np.array(transmission)
    submatrices = np.array(submatrices)
    num_prop = np.array(num_prop)
    params = np.array([a, L, e_zero], dtype=object)
    np.savez(result_file, params=params, energies=energies_global, transmission=transmission, submatrices=submatrices, num_prop=num_prop)
