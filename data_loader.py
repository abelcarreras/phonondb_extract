import numpy as np

from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy import Phonopy
from qha_functions import fit_force_constants, fit_electronic_eos, get_helmoltz_free_energy, get_gibbs_free_energy

from phonopy import file_IO
import yaml
import urllib
import os


################################
# DATABASE URL: http://phonondb.mtl.kyoto-u.ac.jp/database-mp.html
# Structure id
structure_id = 149

# Method of interpolation
interpolation_method = 'quadratic'

# Temperature range
temperatures = np.linspace(0, 1000, 30)

# Number of volumes (to plot the free energy calculated from fitted FC [range will be taken from database])
n_volumes = 30
################################

directory = 'data/mp-{}'.format(structure_id)

if not os.path.isdir(directory):
    try:
        import tarfile
        # Needs: brew install xz
        import backports.lzma as lzma
        print ('downloading data from database')
        urllib.urlretrieve ("http://phonondb.mtl.kyoto-u.ac.jp/_downloads/mp-{}.tar.lzma".format(structure_id), "data/test.tar.lzma")
        tar = tarfile.open(fileobj=lzma.LZMAFile('data/test.tar.lzma'))
        tar.extractall(path='data')
        tar.close()
        os.remove('data/test.tar.lzma')
    except:
        print ('error downloading/extracting file')
        exit()

print 'reading data'

# Read Q-points grid
yaml_file = open(directory+'/quasiharmonic_phonon.yaml', 'r')
qgrid = yaml.load_all(yaml_file).next()['sampling_mesh']


def get_data_from_dir(directory, i_volume):

    data_sets = file_IO.parse_FORCE_SETS(filename=directory+'/phonon-{0:02d}/FORCE_SETS'.format(i_volume))

    yaml_file = open(directory+'/phonon-{0:02d}/phonon.yaml'.format(i_volume), 'r')
    data = yaml.load_all(yaml_file).next()

    unit_cell = PhonopyAtoms(symbols=[item['symbol'] for item in data['points']],
                             scaled_positions=[item['coordinates'] for item in data['points']],
                             cell=data['lattice'])

    phonon = Phonopy(unit_cell, data['supercell_matrix'])

    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()

    force_constants = phonon.get_force_constants()


    supercell = phonon.get_supercell()

    volume = unit_cell.get_volume()
    energy = data['electric_total_energy']

    return supercell, volume, energy, force_constants, data['supercell_matrix']

#force_constants_matrix = np.array([get_force_constants_from_dir(directory, i)[2] for i in range(10)])

force_constants_matrix = []
volumes = []
energies = []
for i in range(10):
    supercell, v, e, fc, scmat = get_data_from_dir(directory, i)
    volumes.append(v)
    energies.append(e)
    force_constants_matrix.append(fc)

force_constants_matrix = np.array(force_constants_matrix)
volumes = np.array(volumes)
energies = np.array(energies)

#supercell data
chemical_symbols = supercell.get_chemical_symbols()
scaled_positions = supercell.get_scaled_positions()
cell = supercell.get_cell()


#Check dimensions
print 'volumes', volumes.shape
print 'energies', energies.shape
print 'force_constants', force_constants_matrix.shape
print 'scaled_positions', scaled_positions.shape
print 'chemical_symbols', len(chemical_symbols)
print 'temperatures', temperatures.shape

primitive_matrix = np.linalg.inv(scmat)


print ('fitting eos')
electronic_eos_f = fit_electronic_eos(volumes, energies)

print ('fitting force constants')
force_constant_f = fit_force_constants(volumes, force_constants_matrix, interpolation=interpolation_method)


# Calculating Helmholtz free energy
fit_volumes = np.linspace(volumes[0], volumes[-1], num=n_volumes)
#fit_volumes = volumes

print ('getting Helmholtz free energy')
free_energies_matrix = []
for t in temperatures:
    print 'temperature', t
    temperature = np.ones_like(fit_volumes) * t
    free_energy_curve = get_helmoltz_free_energy(fit_volumes,
                                                 temperature,
                                                 force_constant_f,
                                                 electronic_eos_f,
                                                 chemical_symbols,
                                                 scaled_positions,
                                                 cell,
                                                 primitive_matrix,
                                                 qgrid)
    free_energies_matrix.append(free_energy_curve)

# Plot Helmholtz free energy
import matplotlib.pyplot as pl

# Data from fited force constants
pl.figure(figsize=(6, 6))
for free_energy_curve in free_energies_matrix:
    pl.plot(fit_volumes, free_energy_curve, marker='', linestyle='-', color='b', label='Fitted FC')
    pl.xlabel('volume (A^3)')
    pl.ylabel('Free energy (eV)')

# Data from Togo's database
hfe_database = np.loadtxt(directory + '/helmholtz-volume.dat', unpack=False).reshape([-1, 10, 2]).swapaxes(1, 2)
for i in temperatures:
    volumes, free_energy_curve = hfe_database[i / 2]
    pl.plot(volumes, free_energy_curve, marker='+', linestyle='', color='r', label='Togo database')


handles, labels = pl.gca().get_legend_handles_labels()
pl.legend([handles[0], handles[-1]], [labels[0], labels[-1]])

pl.savefig('figure-{}.pdf'.format(structure_id), bbox_inches='tight') # Save plot to file
pl.show()