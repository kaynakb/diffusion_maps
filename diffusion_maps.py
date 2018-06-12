# diffusion_maps.py is a compact python module for a modified version of the (an)isotropic
# diffusion maps technique based on the research article of Coifman and Lafon [1].

# [1] R. R. Coifman, S. Lafon, Diffusion maps, Appl. Comput. Harmon. Anal., 21 (2006) 5-30.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from timeit import default_timer as timer
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, svds
from sklearn.metrics.pairwise import _parallel_pairwise, rbf_kernel

__author__ = 'Burak T. Kaynak'
__license__ = 'GPLv3'
__version__ = '0.3'
__email__ = 'kaynakb@gmail.com'


class Diffusion_Maps:
    'An implementation of (an)isotropic diffusion maps technique.'

    __slots__ = [
        '_data', '_alpha', '_time', '_epsilon', '_n_neighbors', '_n_eigvecs',
        '_n_jobs', '_dist', '_kernel_alpha', '_d_alpha', '_eigvals',
        '_eigvecs', '_diffusion_map'
    ]

    def __init__(self,
                 data,
                 alpha=1.0,
                 time=0.0,
                 epsilon=None,
                 n_neighbors=10,
                 n_eigvecs=6,
                 n_jobs=1):
        '''
Docstring:
Diffusion_Maps(data, alpha=1.0, time=0.0, epsilon=None, n_neighbors=10, n_eigvecs=6, n_jobs=1)

Instantiate an object of type Diffusion_Maps.

Parameters
----------
data : array like
     Data
alpha : float, optional (default = 1.0)
      Anisotropic diffusion parameter, which takes values one of the following
      values: 0.0, 1/2, 1.0
time : float, optional (default = None)
     Time steps.
epsilon : float, optional (default = None)
        Scale parameter.
n_neighbors : int, optional (default = 10)
            Number of neighbors to calcualte epsilon.
n_eigvecs : int, optional (default = 6)
          Dimension of the embedding. It should be set to 'all' to calculate all the eigenvectors.
n_jobs : int, optional (default = 1)
       The number of parallel jobs to run when the distance matrix is to be calculated.
If -1, then the number of jobs is set to the number of CPU cores.'''

        self._data = data
        self._alpha = alpha
        self._time = time
        self._epsilon = epsilon
        if epsilon is None:
            self._n_neighbors = n_neighbors
        else:
            self._n_neighbors = None
        self._n_eigvecs = n_eigvecs
        self._n_jobs = n_jobs
        self._dist = None
        self._kernel_alpha = None
        self._d_alpha = None
        self._eigvals = None
        self._eigvecs = None
        self._diffusion_map = None

        self.generate_diffusion_map()

    @property
    def data(self):
        '''
Docstring:
data

Return the data.'''

        return self._data

    @property
    def alpha(self):
        '''
Docstring:
alpha

Return the value of alpha.'''

        return self._alpha

    @property
    def time(self):
        '''
Docstring:
times

Return the value of time.'''

        return self._time

    @property
    def epsilon(self):
        '''
Docstring:
epsilon

Return the value of epsilon.'''

        return self._epsilon

    @property
    def n_neighbors(self):
        '''
Docstring:
n_neighbors

Return the number of neighbors.'''

        return self._n_neighbors

    @property
    def n_eigvecs(self):
        '''
Docstring:
n_eigvecs

Return the number of eigenvectors.'''

        return self._n_eigvecs

    @property
    def kernel(self):
        '''
Docstring:
kernel

Return the renormalized kernel.'''

        return self._kernel_alpha

    @property
    def measure(self):
        '''
Docstring:
measure

Return the Riemannian measure.'''

        return self._d_alpha / np.sum(self._d_alpha)

    @property
    def probability(self):
        '''
Docstring:
probability

Return the transition probability.'''

        return np.diag(1 / self._d_alpha) @ self._kernel_alpha

    @property
    def eigvals(self):
        '''
Docstring:
eigvals

Return the eigenvalues.'''

        return self._eigvals

    @property
    def eigvecs(self):
        '''
Docstring:
eigvecs

Return the renormalized eigenvectors of the diffusion operator.'''

        return self._eigvecs

    @property
    def diffusion_map(self):
        '''
Docstring:
diffusion_map

Return the diffusion map.'''

        return self._diffusion_map

    def __calc_epsilon(self):

        self._dist = _parallel_pairwise(
            self._data, self._data, cdist, n_jobs=self._n_jobs)
        dsn = np.median(np.sort(self._dist, 0)[1:self._n_neighbors + 1], 0)
        self._epsilon = dsn.reshape(-1, 1) * dsn

    def __generate_kernel(self):

        t0 = timer()

        if self._n_neighbors is not None:
            self.__calc_epsilon()
            k_init = np.exp(-self._dist**2 / self._epsilon)
        else:
            k_init = rbf_kernel(self._data, gamma=1 / self._epsilon)
        # k_init = k_init - np.eye(k_init.shape[0])  # prohibits self-transitions

        d_init = np.sum(k_init, 1)
        d_init_alpha = d_init**(-self._alpha)
        d_init_alpha_mat = d_init_alpha.reshape(-1, 1) * d_init_alpha
        self._kernel_alpha = k_init / d_init_alpha_mat
        if np.allclose(self._kernel_alpha, self._kernel_alpha.T):
            pass
        else:
            self._kernel_alpha = (
                self._kernel_alpha + self._kernel_alpha.T) / 2
        self._d_alpha = np.sum(self._kernel_alpha, 1)

        t1 = timer()
        t10 = round(t1 - t0, 3)
        print('time elapsed for the computation of the kernel: {}'.format(t10))

    def __calc_eigs(self):

        if self._n_eigvecs == 'all':

            l, v = eigh(self._kernel_alpha, b=np.diag(self._d_alpha))
            self._eigvals = l[::-1]
            self._eigvecs = v[:, ::-1]
            self._eigvecs = self._eigvecs / self._eigvecs[0, 0]

        else:

            l, v = eigsh(
                self._kernel_alpha,
                k=self._n_eigvecs,
                M=csr_matrix(np.diag(self._d_alpha)))
            self._eigvals = l[::-1]
            self._eigvecs = v[:, ::-1]
            self._eigvecs = self._eigvecs / self._eigvecs[0, 0]

            # The following method can also be used to obtain the diffusion maps.
            # It gives the same result possibly up to a phase.

            # dts = np.diag(self._d_alpha**(-1 / 2))
            # kt = dts @ self._kernel_alpha @ dts
            # if np.allclose(kt, kt.T):
            #     pass
            # else:
            #     kt = (kt + kt.T) / 2

            # u, s, vh = svds(kt, k=self._n_eigvecs)
            # self._eigvals = s[::-1]
            # self._eigvecs = dts @ u[:, ::-1]
            # self._eigvecs = self._eigvecs / self._eigvecs[0, 0]
            # self._diffusion_map = self._eigvecs * self._eigvals**self._time

    def generate_diffusion_map(self,
                               alpha=None,
                               time=None,
                               n_neighbors=None,
                               epsilon=None,
                               n_eigvecs=None):
        '''
Docstring:
generate_diffusion_map(alpha=None, time=None, epsilon=None, n_neighbors=None, n_eigvecs=None)

Generate another diffusion map for the same instance with a different set of parameters.

Parameters
----------
alpha : float, optional
      Anisotropic diffusion parameter, which takes values one of the following
      values: 0.0, 1/2, 1.0
time : float, optional
     Time steps (default None).
epsilon : float, optional
        Scale parameter (default None).
n_neighbors : int, optional
            Number of neighbors (default 10) to calcualte epsilon.
n_eigvecs : int, optional
          Dimension of the embedding, (default 6). It should be set to 'all' to calculate all the eigenvectors.'''

        t0 = timer()

        if self._kernel_alpha is None:
            self.__generate_kernel()
            self.__calc_eigs()

        if (alpha is None) and (epsilon is None) and (n_neighbors is None):
            if n_eigvecs is not None:
                self._n_eigvecs = n_eigvecs
                self.__calc_eigs()
        else:
            if alpha is not None:
                self._alpha = alpha

            if epsilon is not None:
                self._epsilon = epsilon
                self._n_neighbors = None

            if n_neighbors is not None:
                self._n_neighbors = n_neighbors

            self.__generate_kernel()

            if n_eigvecs is not None:
                self._n_eigvecs = n_eigvecs

            self.__calc_eigs()

        if time is not None:
            self._time = time

        self._diffusion_map = self._eigvecs * self._eigvals**self._time

        t1 = timer()
        t10 = round(t1 - t0, 3)
        print('time elapsed for the computation of the diffusion maps: {}'.
              format(t10))
