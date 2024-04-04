import numpy as np    
import numpy.linalg as la
from scipy.linalg.lapack import dtrtri
from scipy.stats import ortho_group, norm, uniform

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from scipy.linalg import sqrtm

def sym(x):
	return 0.5 * (x + x.T)


# The following class describes the setting of SPD manifolds 
class SPD(RiemannianSubmanifold):
    def __init__(self, p, alpha, beta):
        self._p = p
        self._alpha = alpha
        self._beta = beta
        name = f"Manifold of positive definite {p}x{p} matrices"
        dimension = int(p*(p+1)/2)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    # Generation of random pxp SPD matrix given it's conditional number
    def random_point(self, cond=100):
        U = ortho_group.rvs(self._p)
        d = np.zeros(self._p)
        if self._p>2:
	        d[:self._p-2] = uniform.rvs(loc=1/np.sqrt(cond),scale=np.sqrt(cond)-1/np.sqrt(cond),size=self._p-2)
        d[self._p-2] = 1/np.sqrt(cond)
        d[self._p-1] = np.sqrt(cond)
        return U @ np.diag(d) @ U.T

    def random_tangent_vector(self, point):
        return self.projection(point, norm.rvs(size=(self._p,self._p)))

    def zero_vector(self, point):
        return np.zeros((self._p,self._p))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        L = la.cholesky(point)
        iL, _ = dtrtri(L, lower=1)
        coor_a = iL @ tangent_vector_a @ iL.T
        if tangent_vector_a is tangent_vector_b:
            coor_b = coor_a
        else:
            coor_b = iL @ tangent_vector_b @ iL.T
        return self._alpha * np.tensordot(coor_a, coor_b, axes=point.ndim) + self._beta * np.trace(coor_a) * np.trace(coor_b)

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def projection(self, point, vector):
        return sym(np.real(vector))
	
    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return (point @ sym(euclidean_gradient) @ point) / self._alpha - (self._beta / (self._alpha*(self._alpha + self._p * self._beta))) * np.trace(euclidean_gradient @ point) * point

    def retraction(self, point, tangent_vector):
        return np.real(sym( point + tangent_vector + 0.5 * tangent_vector @ la.solve(point, tangent_vector)))

    def transport(self, point_a, point_b, tangent_vector_a):
        tmp = sqrtm(la.solve(point_a, point_b).T) 
        return tmp @ tangent_vector_a @ tmp.T
	
    def dist(self, point_a, point_b):
        L = la.cholesky(point_a)
        iL,_ = dtrtri(L, lower=1)
        tmp = iL @ point_b @ iL.T
        log_eigs = np.log(la.eigh(tmp)[0]) 
        return (self._alpha * np.sum(log_eigs**2) + self._beta * np.sum(log_eigs)**2)**0.5
