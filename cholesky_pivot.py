from numpy import asarray_chkfinite, atleast_2d
from scipy.linalg.decomp import _datacopied
from scipy.linalg import get_lapack_funcs, LinAlgError
import numpy as np

def _cholesky(a, lower=False, overwrite_a=False, clean=True,
              check_finite=True, full_pivot=False, pivot_tol=-1):
    """Common code for cholesky() and cho_factor()."""

    a1 = asarray_chkfinite(a) if check_finite else asarray(a)
    a1 = atleast_2d(a1)

    # Dimension check
    if a1.ndim != 2:
        raise ValueError('Input array needs to be 2 dimensional but received '
                         'a {}d-array.'.format(a1.ndim))
    # Squareness check
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input array is expected to be square but has '
                         'the shape: {}.'.format(a1.shape))

    # Quick return for square empty array
    if a1.size == 0:
        return a1.copy(), lower

    #if not is_hermitian():
    #    raise LinAlgError("Expected symmetric or hermitian matrix")

    overwrite_a = overwrite_a or _datacopied(a1, a)

    # if the pivot flag is false, return the result
    if not full_pivot:
        potrf, = get_lapack_funcs(('potrf',), (a1,))
        c, info = potrf(a1, lower=lower, overwrite_a=overwrite_a, clean=clean)
        if info > 0:
            raise LinAlgError("%d-th leading minor of the array is not positive "
                              "definite" % info)
        if info < 0:
            raise ValueError('LAPACK reported an illegal value in {}-th argument'
                             'on entry to "POTRF".'.format(-info))
        return c, lower
    else: # if the pivot flag is true, return the result plus rank and pivot

        pstrf, = get_lapack_funcs(('pstrf',), (a1,))
        c, pivot, rank, info=pstrf(a1, lower=lower, overwrite_a=overwrite_a, tol=pivot_tol)#

        if info > 0:
            if rank == 0:
                raise LinAlgError("%d-th leading minor of the array is not positive "
                              "semidefinite" % info)
            else:
                raise LinAlgError("The array is rank deficient with "
                              "computed rank %d" % info)

        if info < 0:
            raise ValueError('LAPACK reported an illegal value in {}-th argument'
                         'on entry to "PSTRF".'.format(-info))
        return c, lower, rank, pivot

def cholesky_pivot(a, lower=False, overwrite_a=False, check_finite=True, full_pivot=False, pivot_tol=-1):
    """
    Compute the Cholesky decomposition of a matrix.

    Returns the Cholesky decomposition, :math:`A = L L^*` or
    :math:`A = U^* U` of a Hermitian positive-definite matrix A.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper or lower triangular Cholesky
        factorization.  Default is upper-triangular.
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may improve performance).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    full_pivot : bool, optional
        Whether to use full pivot or not
    pivot_tol : float, optional
        Tolerance for the pivot, if < 0 then tolerance = N*U*MAX( A(M,M) )

    Returns
    -------
    c : (M, M) ndarray
        Upper- or lower-triangular Cholesky factor of `a`.

    Raises
    ------
    LinAlgError : if decomposition fails.

    Examples
    --------
    >>> from scipy.linalg import cholesky
    >>> a = np.array([[1,-2j],[2j,5]])
    >>> L = cholesky(a, lower=True)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> L @ L.T.conj()
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])

    """
    if not full_pivot:
        c,lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True, check_finite=check_finite)
        return c
    else:
        c,lower, rank_bn, piv = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True, check_finite=check_finite, full_pivot=full_pivot, pivot_tol=pivot_tol)
        if lower:
            c = np.tril(c)
        else:
            c = np.triu(c)
        return c, rank_bn, piv
