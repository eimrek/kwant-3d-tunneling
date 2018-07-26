#!/usr/bin/env python

import numpy as np
import time
import matplotlib
matplotlib.use('Agg') # without x environment
import matplotlib.pyplot as plt

import kwant

import os
import argparse
import sys

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

parser.add_argument(
    '--barrier_file',
    metavar='FILENAME',
    required=True,
    help='.npz file containing arrays [box] (nm) and [pot] (eV).')
parser.add_argument(
    '--lat_param',
    metavar='A',
    type=float,
    default=0.05,
    help='Lattice parameter of the tight-binding system in [nm].')
parser.add_argument(
    '--e_min',
    metavar='EMIN',
    type=float,
    required=True,
    help='Minimum energy of electrons.')
parser.add_argument(
    '--e_max',
    metavar='EMAX',
    type=float,
    required=True,
    help='Maximum energy of electrons.')
parser.add_argument(
    '--e_num',
    metavar='E_N',
    type=int,
    required=True,
    help='Number of energies between EMIN and EMAX. This should be a multiple of NCPUS.')
parser.add_argument(
    '--e_leads',
    metavar='E_LEADS',
    type=float,
    default=0.0,
    help='Energy of the leads. This is used to shift everything to lead level.')
parser.add_argument(
    '--out_dir',
    metavar='DIR',
    default="./",
    help='Output directory.')

args = parser.parse_args()
# -----------------------------------------------------------------------------
# Read the barrier data

barrier_data = np.load(args.barrier_file)
box    = barrier_data["box"]
pot    = barrier_data["pot"]
num    = np.array(pot.shape)

e_min = args.e_min
e_max = args.e_max
e_num = args.e_num
e_range_global = np.linspace(e_min, e_max, e_num)

e_leads = args.e_leads
a = args.lat_param

if e_min <= e_leads:
    print("Can't have e_min lower than e_leads.")
    exit(1)

out_dir = args.out_dir
if out_dir[-1] != '/':
    out_dir += '/'

# option to specify non-ground-state modes also in x and y directions
kx_sl = 0.0
ky_sl = 0.0

# -----------------------------------------------------------------------------
# Compare the input potential and the potential on the tight-binding Lattice

num_sparse = (box/a).astype(int)

def potential_at_point(site):
    (x,y,z) = site.pos
    [ix, iy, iz] = np.rint(np.array([x,y,z])/box*(num-1)).astype(int)
    return pot[ix, iy, iz]

pot_sparse = np.empty([num_sparse[0], num_sparse[1], num_sparse[2]])
for i in range(num_sparse[0]):
    for j in range(num_sparse[1]):
        for k in range(num_sparse[2]):
            class site:
                pos = ((i+0.5)*a, (j+0.5)*a, (k+0.5)*a)
            pot_sparse[i, j, k] = potential_at_point(site)

if rank == 0:
    print("Plotting potential slices...")
    f = plt.figure(figsize=(1.2*10,10*box[0]/box[2]))
    cm = plt.pcolormesh(pot[num[0]//2, :, :])
    f.gca().axis('tight')
    f.colorbar(cm)
    plt.savefig(out_dir+"pot_slice_inp.png", dpi=300, bbox_inches='tight')
    plt.close()

    f = plt.figure(figsize=(1.2*10,10*box[0]/box[2]))
    cm = plt.pcolormesh(pot_sparse[num_sparse[0]//2, :, :])
    f.gca().axis('tight')
    f.colorbar(cm)
    plt.savefig(out_dir+"pot_slice_tb.png", dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Divide energy range between the processors
start_i = int(np.round(e_num/size*rank))
if rank == size - 1:
    end_i = e_num
else:
    end_i = int(np.round(e_num/size*(rank+1)))

e_range = e_range_global[start_i:end_i]

# -----------------------------------------------------------------------------
# Kwant setup
L = num_sparse

t = hbar**2/(2*me*a**2) # units: eV

lat = kwant.lattice.general([(a, 0, 0), (0, a, 0), (0, 0, a)])

sys = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((L[0], 0, 0)), lat.vec((0, L[1], 0))))
sys[(lat(i,j,k) for i in range(L[0]) for j in range(L[1]) for k in range(L[2]))] = lambda p: 6*t+potential_at_point(p)
sys[lat.neighbors(1)] = -t

sys = kwant.wraparound.wraparound(sys)

#### Define and attach leads. ####
lead = kwant.Builder(kwant.TranslationalSymmetry((0, 0, -a), lat.vec((L[0], 0, 0)), lat.vec((0, L[1], 0))))
lead[(lat(i, j, 0) for i in range(L[0]) for j in range(L[1]))] = 6 * t + e_leads
lead[lat.neighbors()] = -t
lead = kwant.wraparound.wraparound(lead, keep=0)
sys.attach_lead(lead)
sys.attach_lead(lead.reversed())
# --------------------------------------------------------------- #

if rank == 0:
    print("Plotting kwant system...")
    kwant.plot(sys, site_color=potential_at_point, cmap='jet', site_size=0.4, fig_size=(15, 9), file=(out_dir+"kwant.png"))

sys = sys.finalized()

# -----------------------------------------------------------------------------
# Starting calculation

transmission = []
submatrices = []
num_prop = []

for energy in e_range:
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
    params = np.array([a, L, e_leads], dtype=object)

    mode_0_all_transmission = [(np.abs(submatrices[e])**2)[0].sum() for e in range(len(e_range_global))]

    # plot transmission curve
    f = plt.figure(figsize=(8, 5))
    plt.plot(e_range_global, mode_0_all_transmission, 'o-')
    plt.ylabel("Transmission probability")
    plt.xlabel("Energy (eV)")
    plt.savefig(out_dir+"transmission-prob.png", dpi=300, bbox_inches='tight')
    plt.close()

    f = plt.figure(figsize=(8, 5))
    plt.plot(e_range_global, mode_0_all_transmission, 'o-')
    plt.ylabel("Transmission probability")
    plt.xlabel("Energy (eV)")
    plt.yscale('log')
    plt.savefig(out_dir+"transmission-prob-log.png", dpi=300, bbox_inches='tight')
    plt.close()

    np.savez(out_dir+"results.npz",
             params=params, energies=e_range_global, transmission=transmission,
             submatrices=submatrices, num_prop=num_prop)
