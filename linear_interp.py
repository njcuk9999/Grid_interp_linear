"""
Description of program
"""
import numpy as np
import itertools

# ==============================================================================
# Define variables
# ==============================================================================

# ==============================================================================
# Define functions
# ==============================================================================

class linear_interp(object):
    """
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
    """
    def __init__(self, points, values):
        """
        The data must be defined on a regular grid; the grid spacing however
        may be
        uneven.  Linear and nearest-neighbour interpolation are supported. After
        setting up the interpolator object, the interpolation method (
        *linear* or
        *nearest*) may be chosen at each evaluation.

        Parameters
        ----------
        :param points: tuple of ndarray of float, with shapes
                           (m1, ), ..., (mn, )
                       The points defining the regular grid in n dimensions.

        :param values : array_like, shape (m1, ..., mn, ...)
                        The data on the regular grid in n dimensions.
        """
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self.pdim = values.shape[-1]
        self.ndims = len(points)
        # corresponding indices for combinations
        aindices = [range(len(point)) for point in points]
        permi = np.array(list(itertools.product(*aindices)), dtype=int)
        # find those combinations where we do not have a NaN value
        # and flatten the values to shape of grid
        nanperm = np.zeros(len(permi), dtype=bool)
        # permv = []
        for pi in range(len(permi)):
            nanperm[pi] = np.sum(~np.isnan(values[tuple(permi[pi])])) > 0
        # make non NaN lists
        self.tpermi = [tuple(p) for p in permi[nanperm]]

    def __call__(self, xi):
        """
        This call uses LinearNDInterpolator and the a modified version of
        http://stackoverflow.com/a/14122491 so that we can skip the closest
        grids where all values at that grid point are NaN

        :param xi: parameter list
        :return:
        """
        # work out the closest grid points (lower and upper) to xi
        rawindices = self._find_indices(xi)
        if len(rawindices) == 0:
            return [np.repeat(np.nan, self.pdim)]
        # recalculate the indices for use in next loop
        indices = np.array([j for j in itertools.product(*rawindices)])
        # finally calculate the sub coordinates and interpolate
        sub_coords = []
        for j in xrange(self.ndims):
            sub_coords += [self.grid[j][rawindices[j]]]
        sub_coords = np.array([j for j in itertools.product(*sub_coords)])
        sub_data = self.values[list(np.swapaxes(indices, 0, 1))]
        li = interpolate.LinearNDInterpolator(sub_coords, sub_data)
        return li([xi])

    def _find_indices(self, xi):
        rawindices = []
        for j in xrange(self.ndims):
            idx = np.digitize([xi[j]], self.grid[j])[0]
            rawindices += [[idx - 1, idx]]
        rawindices = np.array(rawindices)
        indices = np.array([j for j in itertools.product(*rawindices)])
        # need to make sure these grid points have values (i.e. not all NaNs)
        # if they don't we need to loop around moving outwards in the grid
        # until we find points that are not NaN
        # if at any point we reach the edge in a certain dimension
        # a NaN array will be returned (because we are out of bounds)
        cond = True
        while cond:
            # need to make sure indices are not NaN values (using tpermi)
            notnan = np.zeros(len(indices), dtype=bool)
            for ii, index in enumerate(indices):
                notnan[ii] = tuple(index) not in self.tpermi
            # if they are all not NaN then we can continue to interpolation
            if np.sum(notnan) == 0:
                cond = False
            else:
                for n in xrange(self.ndims):
                    # the upper and lower bounds in this dimension
                    rawin = rawindices[n]
                    # find the bound where we have NaN values
                    un = np.unique(indices[:, n][notnan]) == rawin
                    # if lower bound then move down an index (unless we are at
                    # the beginning of this dimension then return NaN array
                    # as we are out of bounds
                    if un[0]:
                        if rawin[0] != 0:
                            rawin[0] -= 1
                        else:
                            return []
                    # if upper bound then mvoe up an index (unless we are at the
                    # end of this dimension then return NaN array as we
                    # are out of bounds
                    if un[1]:
                        if rawin[0] != len(self.grid[n]) - 1:
                            rawin[1] += 1
                        else:
                            return []
                # recalculate the indices for use in next loop
                indices = np.array([j for j in itertools.product(*rawindices)])
        # return rawindices
        return rawindices


def find_pos(ps, sgrid):
    indices = [range(i) for i in sgrid.shape[:-1]]
    for i in itertools.product(*indices):
        if tuple(sgrid[tuple(i)]) == tuple(ps):
            return i
    print 'Error pos not found in sgrid'

# ==============================================================================
# Start of code
# ==============================================================================
# test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy import interpolate
    import time

    ipo_new = linear_interp(gridparams.values(), modelgrid)

    ipo_old = interpolate.RegularGridInterpolator(gridparams.values(),
                                                  modelgrid)

    pss = [[5.0, -0.5, 3300.0, 0.2], [5.0, -0.5, 3350.0, 0.2],
           [5.0, -0.5, 3400.0, 0.2], [5.0, -0.5, 3500.0, 0.2],
           [5.0, -0.5, 3550.0, 0.2], [5.0, -0.5, 3600.0, 0.2],
           [5.0, -0.5, 3650.0, 0.2], [5.0, -0.5, 3700.0, 0.2],
           [5.0, -0.5, 3750.0, 0.2], [5.0, -0.5, 3800.0, 0.2],]
    colours = ['r', 'g', 'c', 'b', 'orange', 'purple', '0.5', 'k']*2
    lws = [1]*(len(colours)/2) + [2]*(len(colours)/2)

    plt.close()
    fig, frames = plt.subplots(ncols=1, nrows=2)
    ends1, ends2 = [], []
    for p, ps in enumerate(pss):
        print 'run for ', ps
        # old method
        start = time.time()
        fluxes1 = ipo_old(ps)[0]
        end1 = time.time() - start
        ends1.append(end1)

        kwargs = dict(color=colours[p], lw=lws[p], label=ps)
        if np.sum(~np.isnan(fluxes1)) == 0:
            print 'Old method fails'
        else:
            print 'Old time = {0}'.format(end1)
            frames[0].plot(fluxes1, **kwargs)
            frames[0].legend(loc=0)
        # new method
        start = time.time()
        fluxes2 = ipo_new(ps)
        end2 = time.time() - start
        ends2.append(end2)
        if np.sum(~np.isnan(fluxes2)) == 0:
            print 'New method fails'
        else:
            print 'New time = {0}'.format(end2)
            frames[1].plot(fluxes2, **kwargs)
            frames[1].legend(loc=0)

    frames[0].set_title('Old method  takes {0} seconds'.format(np.mean(ends1)))
    frames[1].set_title('New method  takes {0} seconds'.format(np.mean(ends2)))
    plt.show()
    plt.close()


    # for ps in pss:
    #     fig, frames = plt.subplots(ncols=1, nrows=2)
    #     frames[0].plot(ipo_old(ps)[0], zorder=0, color='k')
    #     frames[0].plot(ipo_new(ps), zorder=1, color='r')
    #     frames[1].plot(ipo_old(ps)[0], zorder=1, color='k')
    #     frames[1].plot(ipo_new(ps), zorder=0, color='r')
    #     plt.suptitle(ps)
    #     plt.show()
    #     plt.close()

# ------------------------------------------------------------------------------

# ==============================================================================
# End of code
# ==============================================================================
