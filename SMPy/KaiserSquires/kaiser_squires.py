import numpy as np

def ks_inversion(g1_grid, g2_grid):
    """
    Perform the Kaiser-Squires inversion to obtain both E-mode and B-mode convergence maps
    from shear components in pixel coordinates, without relying on celestial coordinates.

    :param g1_grid: 2D numpy array of shear component g1.
    :param g2_grid: 2D numpy array of shear component g2.
    :return: Tuple of (kappa_e_grid, kappa_b_grid) as 2D numpy arrays.
    """
    # Get the dimensions of the input grids
    npix_y, npix_x = g1_grid.shape

    # Fourier transform the shear components
    g1_hat = np.fft.fft2(g1_grid)
    g2_hat = np.fft.fft2(g2_grid)

    # Create a grid of wave numbers in units of cycles per pixel
    kx = np.fft.fftfreq(npix_x)  # Frequencies along x-axis
    ky = np.fft.fftfreq(npix_y)  # Frequencies along y-axis
    k1, k2 = np.meshgrid(kx, ky)  # Create 2D grid of frequencies

    # Compute squared magnitude of the wave vector
    k_squared = k1**2 + k2**2

    # Avoid division by zero at the zero frequency component
    k_squared[0, 0] = np.finfo(float).eps

    # Kaiser-Squires inversion in Fourier space
    kappa_e_hat = ((k1**2 - k2**2) * g1_hat + 2 * k1 * k2 * g2_hat) / k_squared
    kappa_b_hat = ((k1**2 - k2**2) * g2_hat - 2 * k1 * k2 * g1_hat) / k_squared

    # Inverse Fourier transform to get the convergence maps
    kappa_e_grid = np.real(np.fft.ifft2(kappa_e_hat))
    kappa_b_grid = np.real(np.fft.ifft2(kappa_b_hat))

    return kappa_e_grid, kappa_b_grid
