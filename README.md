# Grid_interp_linear
This is like the scipy.interpolate.RegularGridInterpolator but usable with an irregular grid

```python
class linear_interp(object):
```
    The data must be defined on a regular grid; the grid spacing however
    may be uneven. After setting up the interpolator object.
    Data may be irregular and filled with NaN (will linearly fit to nearest
    point in that dimension
    Parameters

##### Constructor

    :param points: tuple of ndarray of float, with shapes
                           (m1, ), ..., (mn, )
                       The points defining the regular grid in n dimensions.
    :param values : array_like, shape (m1, ..., mn, ...)
                        The data on the regular grid in n dimensions.


### Example Code

```python

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

```
