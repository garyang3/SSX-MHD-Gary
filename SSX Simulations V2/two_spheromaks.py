import dedalus.public as de
import numpy as np
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt
import os
import h5py
from mpi4py import MPI
from matplotlib.colors import LogNorm

# See "Turbulence analysis of an experimental flux-rope plasma", D A Schaffner et al, 2014.
###########################################################################################
"""
    This scripts contains the two initializations of the spheromaks. spheromak_A is the main 
    and the script that initializes the two spheromaks. 
        There is spheromak_B which appears to be a 1 spheromak initialization.
        getS1 and getS are the two shape functions for the spheromaks.
"""


###########################################################################################
# Shape function
def getS1(r, z, L, R, zCenter):
    #############################################################################################
    """
        This is a script for the shape function described in equation (9) of the above reference.
    """
    #############################################################################################
    lamJ = .1 * L  # Lambda J is a width control variable
    S = np.zeros((r * z).shape)

    r1 = np.copy(r)
    z1 = np.copy(z)
    #################################
    # r - conditions
    #################################
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(r.shape[2]):
                entry = r1[i][j][k]
                if (entry < (R - lamJ)):
                    r1[i][j][k] = 1
                elif (entry <= R and entry >= (R - lamJ)):
                    # r1[i][j][k] = .5*(1-np.cos(np.pi*(R/2-entry)/lamJ))
                    r1[i][j][k] = 0.5 * (1 - np.cos(np.pi * (R - entry) / lamJ))
                elif (entry > R):
                    r1[i][j][k] = 0
                else:
                    r1[i][j][k] = 0.5
                    print("r out of bounds!", r[i][j][k])

    #################################
    # z - conditions
    #################################
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            for k in range(z.shape[2]):
                entry = z1[i][j][k]
                val = -np.abs(entry)
                if (val >= -lamJ and val <= 0):
                    z1[i][j][k] = -0.5 * (1 - np.cos(np.pi * entry / lamJ))
                elif (val > -(L - lamJ) and val < -lamJ):
                    z1[i][j][k] = -1
                elif (val >= -L and val <= -(L - lamJ)):
                    # z1[i][j][k] = -.5*(1-np.cos(np.pi*(L/2+entry)/lamJ))  See getS().
                    z1[i][j][k] = -0.5 * (1 - np.cos(np.pi * (L + entry) / lamJ))
                elif (val < -L):
                    z1[i][j][k] = 0
                else:
                    print("z out of bounds!", entry)
                    z1[i][j][k] = 1

    S = r1 * z1
    # basedir = os.path.dirname(os.path.realpath(__file__))
    # dfile = basedir+'/fields_two/fields_two_s1.h5'
    # data = h5py.File(str(dfile), "r")
    # x = data['scales/x/1.0'][:]
    # y = data['scales/y/1.0'][:]
    # z = data['scales/z/1.0'][:]
    # for i in range (180):
    #    plot_2d(x, y, S[:, :, i], i)
    return S


def plot_2d(x, y, z, i):
    plt.imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap=plt.cm.hot)
    plt.colorbar()
    plt.savefig('2dplots/fig_' + str(i) + '.png')
    plt.close()


def getS(r, z, L, R, zCenter):
    lamJ = .1 * L
    S = np.zeros((r * z).shape)

    r1 = np.copy(r)  # This is Sr in PSST 2014
    z1 = np.copy(z)  # ... in PSST 2014
    """
        When we are moving through cylindrical space.
    """
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(r.shape[2]):

                entry = r1[i][j][k]
                if (entry < (R - lamJ)):  # R is 1, entry is little r. (ref. PSST 2014)
                    r1[i][j][k] = 1
                elif (entry <= R and entry >= (R - lamJ)):
                    # r1[i][j][k] = 0.5*(1 - np.cos(np.pi*(R/2 - entry)/lamJ))  # The function looks different then what is written in PSST. 2014, but it has the same result; ranging from R - lamJ to R.
                    r1[i][j][k] = 0.5 * (1 - np.cos(np.pi * (R - entry) / lamJ))
                    # This part is inverted. Needs to be corrected.
                elif (entry > R):
                    r1[i][j][k] = 0
                else:
                    r1[i][j][k] = 0.5
                    print("r out of bounds!", r[i][j][k])

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            for k in range(z.shape[2]):

                entry = z1[i][j][k]
                val = np.abs(entry)
                if (val <= lamJ - zCenter and val >= 0 - zCenter):
                    z1[i][j][k] = 0.5 * (1 - np.cos(np.pi * entry / lamJ))
                elif (val < (L - lamJ) - zCenter and val > lamJ - zCenter):
                    z1[i][j][k] = 1
                elif (val <= L - zCenter and val >= (L - lamJ) - zCenter):
                    # z1[i][j][k] = 0.5*(1 - np.cos(np.pi*(L/2 - entry)/lamJ))
                    z1[i][j][k] = 0.5 * (1 - np.cos(np.pi * (L - entry) / lamJ))
                elif (val > L - zCenter):
                    z1[i][j][k] = 0
                else:
                    print("z out of bounds!", entry)
                    z1[i][j][k] = 1

    S = r1 * z1
    return S


def spheromak_A(domain, center=(0, 0, 0), B0=1, R=1, L=1):
    """
    This function returns the intial 2X-spheromak vector potential components (x, y, z).
    J0 - Current density
    J1 - Bessel of order 1


    Solve:
    Laplacian(A) = - J0
    J0 = S(r) lam [ -pi J1(a, r) cos(pi z) rhat + lam*J1(a, r)*sin(pi z)     Eq. (9)
    # B0 should always be 1, but we are leaving it as a parameter for safe keeping.
    """

    j1_zero1 = jn_zeros(1, 1)[0]
    kr = j1_zero1 / R
    kz = np.pi / L

    lam = np.sqrt(kr ** 2 + kz ** 2)
    J0 = B0  # This should be 1.
    #####################################################################
    """ Setting up the problem in dedalus. """
    #####################################################################
    #####################################################################
    """ Setting up the domain """
    #####################################################################
    problem = de.LBVP(domain, variables=['Ax', 'Ay', 'Az'])
    #####################################################################
    """ Meta Parameters """
    #####################################################################
    problem.meta['Ax']['y', 'z']['parity'] = -1
    problem.meta['Ax']['x']['parity'] = 1
    problem.meta['Ay']['x', 'z']['parity'] = -1
    problem.meta['Ay']['y']['parity'] = 1
    problem.meta['Az']['x', 'y']['parity'] = -1
    problem.meta['Az']['z']['parity'] = 1
    #####################################################################
    """ Creating fields/variables """
    # Current density components
    #####################################################################
    J0_x = domain.new_field()
    J0_y = domain.new_field()
    J0_z = domain.new_field()
    xx, yy, zz = domain.grids()
    #####################################################################
    """ Setting cylindrical coordinates """
    #####################################################################
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    theta = np.arctan2(yy, xx)
    z = zz - center[2]
    z1 = zz - 10
    #####################################################################
    """ Creating the two shape functions """
    #####################################################################
    S = getS(r, z, L, R, center[2])
    S1 = getS(r, z1, L, R, 10)
    #####################################################################
    """ Current density; cylindrical componetns Eq. (9) """
    # Note they are sums of two seperate shape functions. S and S1.
    # S - centered at 0
    # S1 - centered at 10 (The other end of the domain)
    #####################################################################
    J_r = S * lam * (-np.pi * j1(kr * r) * np.cos(z * kz)) + S1 * lam * (np.pi * j1(kr * r) * np.cos((-z1) * kz))
    J_t = S * lam * (lam * j1(kr * r) * np.sin(z * kz)) - S1 * lam * (-lam * j1(kr * r) * np.sin((-z1) * kz))
    #####################################################################
    """ Initializing the domain fields/grids for the dedalus problem. """
    # J0 is set to B0, which should be 1.
    #####################################################################
    J0_x['g'] = J0 * (J_r * np.cos(theta) - J_t * np.sin(theta))
    J0_y['g'] = J0 * (J_r * np.sin(theta) + J_t * np.cos(theta))
    J0_z['g'] = J0 * S * lam * (kr * j0(kr * r) * np.sin(z * kz)) + J0 * S1 * lam * (
                kr * j0(kr * r) * np.sin((-z1) * kz))

    J0_x.meta['y', 'z']['parity'] = -1
    J0_x.meta['x']['parity'] = 1
    J0_y.meta['x', 'z']['parity'] = -1
    J0_y.meta['y']['parity'] = 1
    J0_z.meta['x', 'y']['parity'] = -1
    J0_z.meta['z']['parity'] = 1

    problem.parameters['J0_x'] = J0_x
    problem.parameters['J0_y'] = J0_y
    problem.parameters['J0_z'] = J0_z

    #####################################################################
    """ Force Free Equations/Spheromak """
    #####################################################################
    # x-component
    problem.add_equation("dx(dx(Ax)) + dy(dy(Ax)) + dz(dz(Ax)) =  -J0_x",
                         condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ax = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")
    # y-component
    problem.add_equation("dx(dx(Ay)) + dy(dy(Ay)) + dz(dz(Ay)) =  -J0_y",
                         condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ay = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")
    # z-component
    problem.add_equation("dx(dx(Az)) + dy(dy(Az)) + dz(dz(Az)) =  -J0_z",
                         condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Az = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    #####################################################################
    """ Building the solver """
    # That is, setting things into play.
    #####################################################################
    solver = problem.build_solver()
    solver.solve()

    return solver.state['Ax']['g'], solver.state['Ay']['g'], solver.state['Az']['g']


def spheromak_B(domain, center=(0, 0, 10), B0=1, R=1, L=1):
    """
        Returns the intial 1X-spheromak vector potential components.
        Solve
        Laplacian(A) = - J0
        J0 = S(r) l_sph [ -pi J1(a r) cos(pi z) rhat + l_sph*J1(a r)*sin(pi z)
        """

    j1_zero1 = jn_zeros(1, 1)[0]
    kr = j1_zero1 / R
    kz = np.pi / L

    lam = np.sqrt(kr ** 2 + kz ** 2)
    J0 = B0

    problem = de.LBVP(domain, variables=['Ax', 'Ay', 'Az'])
    problem.meta['Ax']['y', 'z']['parity'] = -1
    problem.meta['Ax']['x']['parity'] = 1
    problem.meta['Ay']['x', 'z']['parity'] = -1
    problem.meta['Ay']['y']['parity'] = 1
    problem.meta['Az']['x', 'y']['parity'] = -1
    problem.meta['Az']['z']['parity'] = 1

    J0_x = domain.new_field()
    J0_y = domain.new_field()
    J0_z = domain.new_field()
    xx, yy, zz = domain.grids()

    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    theta = np.arctan2(yy, xx)
    z = zz - center[2]

    S = getS1(r, z, L, R, center[2])

    J_r = S * lam * (np.pi * j1(kr * r) * np.cos((-z) * kz))
    J_t = S * lam * (-lam * j1(kr * r) * np.sin((-z) * kz))

    J0_x['g'] = J0 * (J_r * np.cos(theta) - J_t * np.sin(theta))
    J0_y['g'] = J0 * (J_r * np.sin(theta) + J_t * np.cos(theta))
    J0_z['g'] = J0 * S * lam * (kr * j0(kr * r) * np.sin((-z) * kz))

    J0_x.meta['y', 'z']['parity'] = -1
    J0_x.meta['x']['parity'] = 1
    J0_y.meta['x', 'z']['parity'] = -1
    J0_y.meta['y']['parity'] = 1
    J0_z.meta['x', 'y']['parity'] = -1
    J0_z.meta['z']['parity'] = 1

    problem.parameters['J0_x'] = J0_x
    problem.parameters['J0_y'] = J0_y
    problem.parameters['J0_z'] = J0_z

    problem.add_equation("dx(dx(Ax)) + dy(dy(Ax)) + dz(dz(Ax)) =  -J0_x",
                         condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ax = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    problem.add_equation("dx(dx(Ay)) + dy(dy(Ay)) + dz(dz(Ay)) =  -J0_y",
                         condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ay = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    problem.add_equation("dx(dx(Az)) + dy(dy(Az)) + dz(dz(Az)) =  -J0_z",
                         condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Az = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    # Build solver
    solver = problem.build_solver()
    solver.solve()

    return solver.state['Ax']['g'], solver.state['Ay']['g'], solver.state['Az']['g']


def spheromak_1(domain):
    aa_x_1, aa_y_1, aa_z_1 = spheromak_A(domain)
    # aa_x_2, aa_y_2, aa_z_2 = spheromak_B(domain)
    return aa_x_1, aa_y_1, aa_z_1


def spheromak(Bx, By, Bz, domain, center=(0, 0, 0), B0=1, R=1, L=1):
    """
    Incomplete function
    domain must be a dedalus domain
    Bx, By, Bz must be Dedalus fields
        *** center has been changed from (0, 0, 0) to (0, 0, 0.5)
    """

    # parameters
    xx, yy, zz = domain.grids()

    j1_zero1 = jn_zeros(1, 1)[0]
    kr = j1_zero1 / R
    kz = np.pi / L

    lam = np.sqrt(kr ** 2 + kz ** 2)

    # construct cylindrical coordinates centered on center
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    theta = np.arctan2(yy, xx)
    z = zz - center[2]

    # calculate cylindrical fields
    Br = -B0 * kz / kr * j1(kr * r) * np.cos(kz * z)
    Bt = B0 * lam / kr * j1(kr * r) * np.sin(kz * z)

    # convert back to cartesian, place on grid.
    Bx['g'] = Br * np.cos(theta) - Bt * np.sin(theta)
    By['g'] = Br * np.sin(theta) + Bt * np.cos(theta)
    Bz['g'] = B0 * j0(kr * r) * np.sin(kz * z)