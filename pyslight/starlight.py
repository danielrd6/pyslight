#!/usr/bin/env python

import numpy as np
from ifscube.spectools import get_wl
import pyfits as pf
from scipy.interpolate import interp1d
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import scipy.ndimage as ndim
from scipy.ndimage import gaussian_filter as gf


def fits2sl(spec, mask=None, dwl=1, integerwl=True, writetxt=False,
            errfraction=None, normspec=False, normwl=[6100, 6200],
            gauss_convolve=0):
    """
    Converts a 1D FITS spectrum to the format accepted by Starlight.

    Parameters
    ----------
    spec : string
        Name of the FITS spectrum.
    mask : None or string
        Name of the ASCII file containing the regions to be masked,
        as a sequence of initial and final wavelength coordinates, one
        pair per line.
    dwl : number
        The step in wavelength of the resampled spectrum. We recommend
        using the standard 1 angstrom.
    integerwl : boolean
        True if the wavelength coordinates can be written as integers.
    writetxt : boolean
        True if the function should write an output ASCII file.
    errfraction : number
        Fraction of the signal to be used in case the uncertainties
        are unknown.
    gauss_convolve : number
        Sigma of the gaussian kernel to convolve with the spectrum.

    Returns
    -------
    slspec : numpy.ndarray
        2D array with 4 columns, containing wavelength, flux density,
        uncertainty and flags respectively.
    """

    # Loading spectrum from FITS file.
    a = pf.getdata(spec)
    wl = get_wl(spec)

    print('Average dispersion: ', np.average(np.diff(wl)))

    # Linear interpolation of the spectrum and resampling of
    # the spectrum.

    f = interp1d(wl, gf(a, gauss_convolve), kind='linear')
    if integerwl:
        wlrebin = np.arange(int(wl[0]) + 1, int(wl[-1]) - 1)
        frebin = f(wlrebin)

    mcol = np.ones(len(wlrebin))

    if mask is not None:
        masktab = np.loadtxt(mask)
        for i in range(len(masktab)):
            mcol[(wlrebin >= masktab[i, 0]) & (wlrebin <= masktab[i, 1])] = 99

    if normspec:
        normfactor = 1. / np.median(frebin[(wlrebin > normwl[0]) &
                                           (wlrebin < normwl[1])])
    else:
        normfactor = 1.0

    frebin *= normfactor

    if (errfraction is not None) and (mask is not None):
        vectors = [wlrebin, frebin, frebin * errfraction, mcol]
        txt_format = ['%d', '%.6e', '%.6e', '%d']
    elif (errfraction is not None) and (mask is None):
        vectors = [wlrebin, frebin, frebin * errfraction]
        txt_format = ['%d', '%.6e', '%.6e']
    elif (errfraction is None) and (mask is not None):
        vectors = [wlrebin, frebin, mcol]
        txt_format = ['%d', '%.6e', '%d']
    elif (errfraction is None) and (mask is None):
        vectors = [wlrebin, frebin]
        txt_format = ['%d', '%.6e']

    slspec = np.column_stack(vectors)

    if writetxt:
        np.savetxt(spec.strip('fits') + 'txt', slspec, fmt=txt_format)

    return slspec


def readsl(synthfile, full_output=False):

    f = open(synthfile, 'r')
    a = f.readlines()
    f.close()

    skpr = [i for i in np.arange(len(a)) if '## Synthetic spectrum' in a[i]][0]

    b = np.loadtxt(synthfile, skiprows=skpr + 2)

    fobs_norm = float([i.split()[0] for i in a
                       if '[fobs_norm (in input units)]' in i][0])

    b[:, [1, 2]] *= fobs_norm

    return b


def plotsl(synthfile, masked=False, overplot=False):
    """
    Plots the observed spectrum and overlays the resulting SSP
    synthesis.

    Parameters
    ----------
    synthfile : string
        Name of the ASCII file containing the output of Starlight.
    masked : boolean
        Ommit the masked regions from the observed spectrum.

    Returns
    -------
    Nothing.
    """

    b = readsl(synthfile)

    if not overplot:
        fig = plt.figure()
    ax = fig.add_subplot(111)

    if masked:
        m = b[:, 3] > 0
        ax.plot(b[m, 0], b[m, 1], lw=2)

    ax.plot(b[:, 0], b[:, 1])
    ax.plot(b[:, 0], b[:, 2])

    plt.show()

    return


def subtractmodel(synthfile, fitsfile=None, writefits=False):

    a = readsl(synthfile)

    if fitsfile is None:
        b = np.column_stack([a[:, 0], a[:, 1] - a[:, 2]])
    else:
        wl = get_wl(fitsfile)
        f = interp1d(wl, pf.getdata(fitsfile))
        m = interp1d(a[:, 0], a[:, 2], bounds_error=False, fill_value=0)
        b = np.column_stack([wl, f(wl) - m(wl)])

    if writefits:

        if fitsfile is None:
            print('ERROR! No FITS file given.')
            return

        pf.writeto(fitsfile[:-4] + 'sp.fits', f(wl) - m(wl),
                   header=pf.getheader(fitsfile))

    return b


def powerlaw_flux(synthfile, wl=5100, alpha=0.5):

    with open(synthfile, 'r') as f:
        synth = f.readlines()

    wln = float(synth[22].split()[0])
    fnorm = float(synth[25].split()[0])
    xpl = float(synth[108].split()[1])

    def powerlaw(wl, wlnorm=4620, alpha=0.5):
        return (wl / float(wlnorm)) ** (-1 - alpha)

    print(wln, fnorm, xpl)

    f_lambda = fnorm * xpl / 100. * powerlaw(wl)

    return f_lambda
