import numpy as np

def ks_inversion(g1_grid, g2_grid):
    """
    Perform the Kaiser-Squires inversion to obtain the convergence map from shear components.
    """
    # Get the dimensions of the input grids
    npix_dec, npix_ra = g1_grid.shape

    # Fourier transform the shear components
    g1_hat = np.fft.fft2(g1_grid)
    g2_hat = np.fft.fft2(g2_grid)

    # Create a grid of wave numbers
    k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
    k_squared = k1**2 + k2**2

    # Kaiser-Squires inversion in Fourier space
    # Avoid division by zero by setting zero frequency component to zero
    kappa_hat = np.where(k_squared != 0, (1 / k_squared) * ((k1**2 - k2**2) * g1_hat + 2 * k1 * k2 * g2_hat), 0)

    # Inverse Fourier transform to get the convergence map
    kappa_grid = np.fft.ifft2(kappa_hat)

    return np.real(kappa_grid)  # Return the real part