import numpy as np

def ks_inversion(g1_grid, g2_grid, npix):
    """
    Perform the Kaiser-Squires inversion to obtain the convergence map from shear components.

    Parameters
    ----------
    g1_grid, g2_grid : ndarray
        2D numpy arrays of the binned g1 and g2 shear values.
    npix : int
        Number of pixels (bins) along each axis of the grid.

    Returns
    -------
    kappa_grid : ndarray
        2D numpy array of the convergence (kappa) values. 
    """

    # Fourier transform the shear components directly
    g1_hat = np.fft.fft2(g1_grid)
    g2_hat = np.fft.fft2(g2_grid)

    # Create a grid of wave numbers
    k1, k2 = np.meshgrid(np.fft.fftfreq(npix), np.fft.fftfreq(npix))
    k_squared = k1**2 + k2**2
    k_squared[0, 0] = 1  # Avoid division by zero at the zero frequency

    # Kaiser-Squires inversion in Fourier space
    kappa_hat = (g1_hat * k1**2 + g2_hat * k1 * k2) / k_squared

    # Inverse Fourier transform to get the convergence map
    kappa_grid = np.fft.ifft2(kappa_hat)

    return np.real(kappa_grid)  # Return the real part
