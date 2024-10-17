import numpy as np

def ks_inversion(g1_grid, g2_grid):
    """
    Perform the Kaiser-Squires inversion to obtain both E-mode and B-mode convergence maps from shear components.
    """
    # Get the dimensions of the input grids
    npix_dec, npix_ra = g1_grid.shape

    # Fourier transform the shear components
    g1_hat = np.fft.fft2(g1_grid)
    g2_hat = np.fft.fft2(g2_grid)

    # Create a grid of wave numbers
    k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
    k_squared = k1**2 + k2**2

    # Avoid division by zero by replacing zero values with a small number
    k_squared = np.where(k_squared == 0, np.finfo(float).eps, k_squared)

    # Kaiser-Squires inversion in Fourier space
    kappa_e_hat = (1 / k_squared) * ((k1**2 - k2**2) * g1_hat + 2 * k1 * k2 * g2_hat)
    kappa_b_hat = (1 / k_squared) * ((k1**2 - k2**2) * g2_hat - 2 * k1 * k2 * g1_hat)

    # Inverse Fourier transform to get the convergence maps
    kappa_e_grid = np.real(np.fft.ifft2(kappa_e_hat))
    kappa_b_grid = np.real(np.fft.ifft2(kappa_b_hat))

    return kappa_e_grid, kappa_b_grid
