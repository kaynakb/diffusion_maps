# diffusion_maps.py is a compact python module for the (an)isotropic diffusion
# maps technique based on the research article of Coifman and Lafon [1].

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
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, svds
from sklearn.metrics.pairwise import rbf_kernel

__author__ = 'Burak T. Kaynak'
__license__ = 'GPLv3'
__version__ = '0.1'
__email__ = 'kaynakb@gmail.com'


class Diffusion_Maps:
    '''An implementation of (an)isotropic diffusion maps, originally written by Coifman and Lafon [1].

       [1] R. R. Coifman, S. Lafon, Diffusion maps, Appl. Comput. Harmon. Anal., 21 (2006) 5-30.'''

    __slots__ = [
        '_data', '_alpha', '_epsilon', '_time', '_n_eigvecs', '_kernel_alpha',
        '_d_alpha', '_eigvals', '_eigvecs', '_diffusion_maps'
    ]

    def __init__(self,
                 data,
                 alpha=1.0,
                 epsilon=None,
                 time=0.0,
                 n_eigvecs='all'):
        '''
Docstring:
Diffusion_Maps(data, alpha=1.0, epsilon=None, time=0.0, n_eigvecs='all')

Creates a Diffusion_Maps instance.

Parameters
----------
data : array like
     Data
alpha : float, optional
      Anisotropic diffusion parameter, which takes values one of the following
      values: 0.0, 1/2, 1.0
epsilon : float, optional
        Scale parameter, whose deafult value is None that corresponds to
        the dimension of the feature space.
time : float, optional
     Time steps (default None).
n_eigvecs : int, optional
          Dimension of the embedding, (default 'all').'''

        self._data = data
        self._alpha = alpha
        self._epsilon = float(
            self._data.shape[1]) if epsilon is None else epsilon
        self._time = time
        self._n_eigvecs = n_eigvecs

        self.generate_diffusion_maps(self._alpha, self._epsilon, self._time,
                                     self._n_eigvecs)

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
    def epsilon(self):
        '''
Docstring:
epsilon

Return the value of epsilon.'''

        return self._epsilon

    @property
    def time(self):
        '''
Docstring:
times

Return the value of time.'''

        return self._time

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
    def diffusion_maps(self):
        '''
Docstring:
diffusion_maps

Return the diffusion map.'''

        return self._diffusion_maps

    def __generate_kernel(self):

        t0 = timer()

        k_init = rbf_kernel(self._data, gamma=1 / self._epsilon)
        # k_init = k_init - np.eye(k_init.shape[0])  # prohibits self-transitions
        d_init = np.sum(k_init, 1)
        d_init_alp_inv = np.diag(d_init**(-self._alpha))
        self._kernel_alpha = d_init_alp_inv @ k_init @ d_init_alp_inv
        if np.allclose(self._kernel_alpha, self._kernel_alpha.T):
            pass
        else:
            self._kernel_alpha = (
                self._kernel_alpha + self._kernel_alpha.T) / 2
        self._d_alpha = np.sum(self._kernel_alpha, 1)

        t1 = timer()
        t10 = round(t1 - t0, 3)
        print('time elapsed for the computation of the kernel: {}'.format(t10))

    def generate_diffusion_maps(self,
                                alpha=None,
                                epsilon=None,
                                time=None,
                                n_eigvecs=None):
        '''
Docstring:
generate_diffusion_maps(alpha=None, epsilon=None, time=None, n_eigvecs=None)

Generate another diffusion map for the same instance with a different set of parameters.

Parameters
----------
alpha : float, optional
      Anisotropic diffusion parameter, which takes values one of the following
      values: 0.0, 1/2, 1.0
epsilon : float, optional
        Scale parameter, whose deafult value is None that corresponds to
        the dimension of the feature space.
time : float, optional
     Time steps (default None).
n_eigvecs : int, optional
          Dimension of the embedding, (default 'all').'''

        t0 = timer()

        if alpha is not None:
            self._alpha = alpha

        if epsilon is not None:
            self._epsilon = epsilon

        if time is not None:
            self._time = time

        if n_eigvecs is not None:
            self._n_eigvecs = n_eigvecs

        self.__generate_kernel()

        if self._n_eigvecs == 'all':

            l, v = eigh(self._kernel_alpha, b=np.diag(self._d_alpha))
            self._eigvals = l[::-1]
            self._eigvecs = v[:, ::-1]
            self._eigvecs = self._eigvecs / self._eigvecs[0, 0]
            self._diffusion_maps = self._eigvecs * self._eigvals**self._time

        else:

            l, v = eigsh(
                self._kernel_alpha,
                k=self._n_eigvecs,
                M=csr_matrix(np.diag(self._d_alpha)))
            self._eigvals = l[::-1]
            self._eigvecs = v[:, ::-1]
            self._eigvecs = self._eigvecs / self._eigvecs[0, 0]
            self._diffusion_maps = self._eigvecs * self._eigvals**self._time

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
            # self._diffusion_maps = self._eigvecs * self._eigvals**self._time

        t1 = timer()
        t10 = round(t1 - t0, 3)
        print('time elapsed for the computation of the diffusion maps: {}'.
              format(t10))
