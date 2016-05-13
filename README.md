# Grid_interp_linear
This is like the scipy.interpolate.RegularGridInterpolator but usable with an irregular grid


    The data must be defined on a regular grid; the grid spacing however
    may be uneven. After setting up the interpolator object.
    
    Data may be irregular and filled with NaN (will linearly fit to nearest
    point in that dimension

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
