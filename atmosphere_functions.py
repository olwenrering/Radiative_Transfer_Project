import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy import constants as const
import os


def partition_function(energy_levels, degeneracies, T_atm):
    """
    Determines the value of the partition function of Boltzmann distribution 
    (assuming constant temperature in the atmosphere)
    
    :param energy_levels: retrieved energy levels for the species (in J) 
    :type energy_levels: array

    :param degeneracies: degeneracies of the retrieved levels (dimensionless) 
    :type degeneracies: array

    :param T_atm: temperature of the atmosphere assumed constant (in K) 
    :type T_atm: float

    :return: Z
    :rtype: float
    """
    
    M = len(energy_levels)
    Z = 0
    
    for i in range(M):
        Z += degeneracies[i] * np.exp(-energy_levels[i]/(const.k_B.value * T_atm))
    
    return Z


def energy_level_density(energy_level,degeneracy,T_atm, Z, n_tot):
    """
    Determines the density at height z for a certain level given its energy and degeneracy
    (assuming constant temperature in the atmosphere)
    
    :param energy_level: energy level (in J) 
    :type energy_level: float

    :param degeneracy: degeneracy of the level (dimensionless) 
    :type degeneracy: float

    :param T_atm: temperature of the atmosphere assumed constant (in K) 
    :type T_atm: float
    
    :param Z: value of partition function (dimensionless) 
    :type Z: float
    
    :param n_tot: total density at height z (m-3)
    :type n_tot: float
    
    :return: n_level
    :rtype: float
    """
    
    Numerator = degeneracy * np.exp(-energy_level/(const.k_B.value * T_atm))
    P = Numerator / Z
    
    n_level = n_tot * P
    
    return n_level


def surface_gravity(M_p, R_p):
    """
    Determines surface gravity of the planet (in ms-2)

    :param M_p: mass of the planet (in kg) 
    :type M_p: float

    :param R_p: radius of the planet (in m) 
    :type R_p: float

    :return: g
    :rtype: float
    
    """

    g = const.G.value * M_p / R_p**2

    return g

def scale_height(T_atm, Mm, M_p, R_p):
    """
    Determines the scale height of the atmosphere (in m) 

    :param T_atm: temperature of the atmosphere (in K) 
    :type T_atm: float

    :param Mm: molecular mass (g/mol)
    :type Mm: float

    :param M_p: mass of the planet (in kg) 
    :type M_p: float

    :param R_p: radius of the planet (in m) 
    :type R_p: float

    :return: H
    :rtype: float
    """
    g = surface_gravity(M_p, R_p)
    H = (const.k_B.value * T_atm) / (Mm*const.m_p.value * g)

    return H

def surface_number_density(P0, T_atm):
    """
    Determines surface number density with ideal gas law (in m-3)

    :param P0: pressure of the atmosphere (in Pa) 
    :type P0: float

    :param T_atm: temperature of the atmosphere (in K) 
    :type T_atm: float

    :return: n0 the number density (molecules/m^3)
    :rtype: float
    """
    n0 = (P0 / (const.k_B.value * T_atm))

    return n0

def absorption_cross_section(centre, width, sigma0, wavelength):
    """
    Determines absorption cross section assuming a gaussian centered at a particular wavelength (in m2) 

    :param centre: central wavelength (in microns) 
    :type centre: float

    :param width: width of the gaussian (dimensionless)
    :type width: float

    :param sigma0: value for the central absorption cross section (in m2)
    :type sigma0: float

    :param wavelength: wavelength (in microns)
    :type width: float
    
    :return: sigma_lambda
    :rtype: float
    """
    sigma_lambda = sigma0 * np.exp(- ((wavelength - centre) / width)**2)

    return sigma_lambda

def altitude_tau_1(sigma_lambda, n0, R_p, H):
    """
    Determines the altitude z where tau is 1 (in m)

    :param sigma_lambda: absorption cross section (m2)
    :type centre: float

    :param n0: surface number density with ideal gas law (in m-3)
    :type width: float

    :param R_p: radius of the planet (in m) 
    :type R_p: float

    :param H: scale height of the atmosphere (in m) 
    :type width: float
    
    :return: z_tau1
    :rtype: float
    """
    
    inside_log = (sigma_lambda * n0 * np.sqrt(2 * np.pi * R_p * H))
    #Important note, if the value inside_log is smaller than one, then the value of altitude_tau_1 would be negative
    #We could study that case separatly and just say that in that case the atmosphere is optically thin and make z=0 for example
    #I am not including that case in this first version. 
    z_tau1 = H * np.log(inside_log)

    return z_tau1

def eff_radius_planet(R_p, z_tau1):
    """
    Determines the effective radius of the planet with atmosphere (in m) 

    :param R_p: radius of the planet (in m) 
    :type R_p: float

    :param z_tau1: the altitude z where tau is 1 (in m) 
    :type z_tau1: float

    :return: R_eff
    :rtype: float
    
    """
    
    R_eff = R_p + z_tau1

    return R_eff
    
def transit_depth_lambda(R_eff, R_star):
    """
    Determines the wavelength-dependent depth of an exoplanet transit.

    :param R_eff: the effective radius of the planet with atmosphere (in m)
    :type Rp: float

    :param R_star: the radius of the star (in m)
    :type R_star: float

    :return: delta_lambda
    :rtype: float
    """

    delta_lambda = (R_eff / R_star)**2
    return delta_lambda


def density_profile_atm_consT(z, T_atm, H, P0):
    """
    Determines the density profile (in m-3) of a certain species in the atmosphere based on simple hydrostatic equilibrium + ideal gas law.
    This function assumes constant temperature T(z)=T_atm.

    :param z: height above the surface (in m)
    :type z: float
    
    :param T_atm: temperature of the atmosphere assumed constant (in K) 
    :type T_atm: float

    :param Mm: molecular mass (g/mol)
    :type Mm: float


    :param M_p: mass of the planet (in kg) 
    :type M_p: float

    :param R_p: radius of the planet (in m) 
    :type R_p: float

    :param P0: pressure of the atmosphere (in Pa) 
    :type P0: float

    :return: n
    :rtype: float
    
    """
    n0 = surface_number_density(P0, T_atm)

    n = n0 * np.exp(-z/H)

    return n

def density_profile_atm_linearT(z, T_atm_0, Mm, M_p, R_p, P0, constant):
    """
    Determines the density profile (in m-3) of a certain species in the atmosphere based on simple hydrostatic equilibrium + ideal gas law.
    This function assumes a temperature T(z) = T_atm_0 - constant * z. 

    :param T_atm_0: temperature of the atmosphere at the surface (in K) 
    :type T_atm_0: float

    :param Mm: molecular mass (g/mol)
    :type Mm: float

    :param M_p: mass of the planet (in kg) 
    :type M_p: float

    :param R_p: radius of the planet (in m) 
    :type R_p: float

    :param P0: pressure of the atmosphere (in Pa) 
    :type P0: float

    :param constant: constant in the linear expression (in K/m)
    :type constnat: float

    :return: n
    :rtype: float
    
    """

    T = T_atm_0 - constant * z 
    g = surface_gravity(M_p, R_p)
    n0 = surface_number_density(P0, T_atm_0)

    n = (P0 / (const.k_B.value * T)) * (1 - constant * z / T_atm_0) ** ((g * (Mm/(1000 * const.N_A.value))) / (constant * const.k_B.value))

    return n




def fast_binning(x, y, bins, error=None, std=False):
    bins = np.arange(np.min(x), np.max(x), bins)
    d = np.digitize(x, bins)

    n = np.max(d) + 2

    binned_x = np.empty(n)
    binned_y = np.empty(n)
    binned_error = np.empty(n)

    binned_x[:] = -np.pi
    binned_y[:] = -np.pi
    binned_error[:] = -np.pi

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            binned_y[i] = np.mean(y[s])
            binned_x[i] = np.mean(x[s])
            binned_error[i] = np.std(y[s]) / np.sqrt(len(s))

            if error is not None:
                err = error[s]
                binned_error[i] = np.sqrt(np.sum(np.power(err, 2))) / len(err)
            else:
                binned_error[i] = np.std(y[s]) / np.sqrt(len(s))

    nans = binned_x == -np.pi
    
    return binned_x[~nans], binned_y[~nans], binned_error[~nans]