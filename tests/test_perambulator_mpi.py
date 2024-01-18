import os
import sys
import numpy
# mpi gather lattice data and save
from pyquda.core import gatherLattice
from pyquda import getMPIRank
from pyquda import enum_quda
import re
import time
from time import perf_counter

from lattice import PerambulatorGenerator, PerambulatorNpy
from lattice import set_backend, get_backend, check_QUDA
from lattice import GaugeFieldIldg, EigenvectorNpy, Nc, Nd
# test_dir = os.path.dirname(os.path.abspath(__file__))
# print(test_dir)
# sys.path.insert(0, os.path.join(test_dir, ".."))

conf_id = sys.argv[1]
conf_spec=sys.argv[2]
savePath = F"./result/{conf_id}"
gaugePath = F"/public/group/lqcd/configurations/CLOVER/{conf_spec}/{conf_spec}_cfg_{conf_id}.lime"
evecPath = F"/public/group/lqcd/eigensystem/{conf_spec}/{conf_id}"
clover_coeff = 1.160920226

pattern_Nt = r"x(\d+)$" # pattern to match a digit (\d) one or more times (+) after an x and at the end of the string ($)
match = re.search(pattern_Nt, conf_spec) # search for match
if match: # if match is found
    number = match.group(1) # get the first group (the part inside parentheses)
    Nt = int(number)
    print(number) # print result
else: # if no match is found
    print("No match") # print message

pattern_Nx = r"_L(\d+)x" # pattern to match a digit (\d) one or more times (+) after an x and at the end of the string ($)
match = re.search(pattern_Nx, conf_spec) # search for match
if match: # if match is found
    number = match.group(1) # get the first group (the part inside parentheses)
    Nx = int(number)
    print(number) # print result
else: # if no match is found
    print("No match") # print message

grid_size = [1, 1, 1, 2]
if not check_QUDA(grid_size):
    raise ImportError("Please install PyQuda")
latt_size = [Nx, Nx, Nx, Nt]t
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
Lx, Ly, Lz, Lt = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Ne = 100
Ns = 4

set_backend("cupy")
backend = get_backend()

gauge_field = GaugeFieldIldg(f"{gaugePath}/", ".lime", [Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc])
eigenvector = EigenvectorBinary(f"{evecPath}/", [Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc], Ne)
perambulator = PerambulatorGenerator(
    latt_size, gauge_field, eigenvector, -0.277, 1e-7, 5000, 1.0, 1.0, clover_coeff, clover_coeff, True, [[4, 4, 3, 3], [2, 2, 2, 2], [4, 4, 4, 4]]
)  # arbitrary dslash parameters
perambulator.dslash.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SUMMARIZE

peramb = backend.zeros((Lt * Gt, Lt, Ns, Ns, Ne, Ne), "<c16")

perambulator.load(F"{conf_spec}_cfg_{conf_id}")
perambulator.stout_smear(1, 0.125)
for t in range(Lt * Gt):
    peramb[t] = perambulator.calc(t)

    # Note: For perambulator, mpi gather always gather timeslices and reduce space!
    peramb_h = gatherLattice(peramb[t].get(), axes = [0, -1, -1, -1], reduce_op="sum", root=0)
    if getMPIRank() == 0:
        # for t in range(Lt * Gt):
        #     peramb_h[t] = numpy.roll(peramb_h[t], -t, 0)
        # numpy.save(f"{out_prefix}{cfg}{out_suffix}", peramb_h)
        peramb_h = peramb_h.transpose(2, 0, 4, 1, 3) #d_source, t_sink, ev_source, d_sink, ev_sink
        # print(perambulator_save_tsrc.shape)
        s = perf_counter()
        for d_source in range(0,4):
            peramb_h[d_source].tofile(F"{savePath}/perams.{conf_id}.{d_source}.{t}")
        print(
            FR"SAVE: {perf_counter()-s:.3f} sec for perambulator timeslice {t}."
        )

    # check data
    # check(cfg, peramb)

perambulator.dslash.destroy()
