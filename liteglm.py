import numpy as np
import pandas as pd
from statsmodels.genmod.families.family import Binomial
from statsmodels.genmod.families.links import Logit
from scipy.linalg import svd, solve_triangular, qr, cholesky
import patsy
from cholesky_pivot import cholesky_pivot

def glm_svd_newton(data, formula, y, family=Binomial(), link=Logit(), maxit=25, tol=1e-08,stol=1e-08,singular_ok=True,
                      weights=None, reg_method= "column projection"):
    X = patsy.dmatrix(formula, data=data, return_type='dataframe')
    S = list(svd(X)) # S[0]=u S[1]=d S[2]=v
    V=S[2]
    nVars=S[2].shape[1]
    idx=np.arange(nVars)
    i = (S[1] / S[1][0]) > stol
    k = np.sum(i)
    pivot=np.arange(nVars)
    if (k < nVars):
        if reg_method == "column projection":
            Q,R,pivot=qr(S[2][:, :k], pivoting=True)
            idx=np.argsort(pivot[:k])
            omit=pivot[-(nvars-k):]
            S_new = svd(X[~idx.isin(omit)])
            if((S_new[-1] / S_new[0]) <= stol):
                print("Whoops! SVD subset selection failed, trying dqrdc2 on full matrix")
            if(len(X)==3):
                Q = np.matmul(S[2], S[1] + S[0].T)
            else:
                Q,R,pivot=qr(X, pivoting=True)
            pivot = Q[2]
            idx=np.argsort(pivot[:k])
            omit=pivot[-(nvars-k):]
            S_new = svd(X[~idx.isin(omit)])
        S = S_new
        print("omitting column(s) ", omit)

    s = np.zeros(nVars)
    nobs=y.shape[0]
    nVars=S[2].shape[1]
    if weights is None:
        weights =np.ones(nobs)
    varianceFam=family.variance
    linkinvFam=link.inverse
    mu_eta=link.inverse_deriv
    etastart = None
    if len(y.shape)==1:
        mustart = (weights * y + 0.5)/(weights + 1)
    else:
        n = y + weights
        ytmp = 0 if not n else y[:, 1]/n
        mustart = (n * ytmp + 0.5)/(n + 1)
    eta=link(mustart)
    dev_resids = lambda y,m ,w : family.resid_dev(y,m,w)**2
    dev = sum(dev_resids(y, linkinvFam(eta), weights))
    devold = 0
    for j in range(maxit):
        g      = linkinvFam(eta)
        varg   = varianceFam(g)
        if (np.any(np.isnan(varg))): raise LinAlgError("NAs in variance of the inverse link function")
        if (np.any(varg==0)): raise LinAlgError("Zero value in variance of the inverse link function")
        gprime  = mu_eta(eta)
        if (np.any(np.isnan(gprime))): raise LinAlgError("NAs in the inverse link function derivative")
        z = eta + (y - g) / gprime
        W = weights * (gprime**2 / varg)
        W=W.reshape(len(W),1)
        cross1=np.matmul(np.transpose(S[0][:,:7]), W * S[0][:,:7])
        C, rank_bn, piv = cholesky_pivot(cross1, full_pivot=True)
        cross2=np.matmul(np.transpose(np.asarray(S[0][:,:7])), np.asarray(W.reshape(-1)*z))[piv-1]
        s = solve_triangular(np.transpose(C), cross2, lower=True)
        s = solve_triangular(C, s, lower=False)[np.argsort(piv)]
        eta = np.matmul(S[0][:,:7], s)
        dev = np.sum(dev_resids(y, g, weights))
        if(np.absolute(dev - devold) / (0.1 + np.absolute(dev)) < tol): break
        devold = dev
    x = np.empty(X.shape[1])
    x[:] = np.nan
    inV=1/S[1]
    if(reg_method == "minimum norm"): inV[inV > 1/stol] = 1
    x[idx] = np.matmul(S[2].T, (s*inV).reshape(-1,1)).reshape(-1)
    return(x, j+1, k, pivot) # coefficients=x,iterations=j, rank=k, pivot=pivot

def glm_svd_newton_dm(X, y, family=Binomial(), link=Logit(), maxit=25, tol=1e-08,stol=1e-08,singular_ok=True,
                      weights=None, reg_method= "column projection"):
    S = list(svd(X)) # S[0]=u S[1]=d S[2]=v
    V=S[2]
    nVars=S[2].shape[1]
    idx=np.arange(nVars)
    i = (S[1] / S[1][0]) > stol
    k = np.sum(i)
    pivot=np.arange(nVars)
    if (k < nVars):
        if reg_method == "column projection":
            Q,R,pivot=qr(S[2][:, :k], pivoting=True)
            idx=np.argsort(pivot[:k])
            omit=pivot[-(nvars-k):]
            S_new = svd(X[~idx.isin(omit)])
            if((S_new[-1] / S_new[0]) <= stol):
                print("Whoops! SVD subset selection failed, trying dqrdc2 on full matrix")
            if(len(X)==3):
                Q = np.matmul(S[2], S[1] + S[0].T)
            else:
                Q,R,pivot=qr(X, pivoting=True)
            pivot = Q[2]
            idx=np.argsort(pivot[:k])
            omit=pivot[-(nvars-k):]
            S_new = svd(X[~idx.isin(omit)])
        S = S_new
        print("omitting column(s) ", omit)

    s = np.zeros(nVars)
    nobs=y.shape[0]
    nVars=S[2].shape[1]
    if weights is None:
        weights =np.ones(nobs)
    varianceFam=family.variance
    linkinvFam=link.inverse
    mu_eta=link.inverse_deriv
    etastart = None
    if len(y.shape)==1:
        mustart = (weights * y + 0.5)/(weights + 1)
    else:
        n = y + weights
        ytmp = 0 if not n else y[:, 1]/n
        mustart = (n * ytmp + 0.5)/(n + 1)
    eta=link(mustart)
    dev_resids = lambda y,m ,w : family.resid_dev(y,m,w)**2
    dev = sum(dev_resids(y, linkinvFam(eta), weights))
    devold = 0
    for j in range(maxit):
        g      = linkinvFam(eta)
        varg   = varianceFam(g)
        if (np.any(np.isnan(varg))): raise LinAlgError("NAs in variance of the inverse link function")
        if (np.any(varg==0)): raise LinAlgError("Zero value in variance of the inverse link function")
        gprime  = mu_eta(eta)
        if (np.any(np.isnan(gprime))): raise LinAlgError("NAs in the inverse link function derivative")
        z = eta + (y - g) / gprime
        W = weights * (gprime**2 / varg)
        W=W.reshape(len(W),1)
        cross1=np.matmul(np.transpose(S[0][:,:7]), W * S[0][:,:7])
        C, rank_bn, piv = cholesky_pivot(cross1, full_pivot=True)
        cross2=np.matmul(np.transpose(np.asarray(S[0][:,:7])), np.asarray(W.reshape(-1)*z))[piv-1]
        s = solve_triangular(np.transpose(C), cross2, lower=True)
        s = solve_triangular(C, s, lower=False)[np.argsort(piv)]
        eta = np.matmul(S[0][:,:7], s)
        dev = np.sum(dev_resids(y, g, weights))
        if(np.absolute(dev - devold) / (0.1 + np.absolute(dev)) < tol): break
        devold = dev
    x = np.empty(X.shape[1])
    x[:] = np.nan
    inV=1/S[1]
    if(reg_method == "minimum norm"): inV[inV > 1/stol] = 1
    x[idx] = np.matmul(S[2].T, (s*inV).reshape(-1,1)).reshape(-1)
    return(x, j+1, k, pivot) # coefficients=x,iterations=j, rank=k, pivot=pivot

def chunck_generator(filename, header,chunk_size ):
   for chunk in pd.read_csv(filename,delimiter=',', iterator=True,header=header, 
                        chunksize=chunk_size, parse_dates=[1]): 
        yield (chunk)

def _generator( filename, header=None,chunk_size = 100):
    chunk = chunck_generator(filename, header,chunk_size)
    for row in chunk:
        yield row

def irls_incremental_dm(filename, chunksize, b, family=Binomial(), link=Logit(), maxit=25, tol=1e-08, header=None, 
                     headerNames=None):
    # filename contains the data after the patsy formula is applied
    x=None
    nRows=chunksize
    tmp=pd.read_csv(filename,delimiter=',', header=None,nrows=1, parse_dates=[1])
    nCols=tmp.shape[1]
    for j in range(1, maxit+1):
        k=0
        generator = _generator(filename=filename, header=header,chunk_size = nRows)
        if x is None: x=np.zeros(nCols)
        ATWA = np.zeros((nCols, nCols))
        ATWz = np.zeros(nCols)
        for rowA in generator:
            A=np.asarray(rowA, dtype=np.float32)
            eta = np.matmul(A, x)
            eta=eta.reshape(len(eta))
            g = link.inverse(eta)
            mu_eta=link.inverse_deriv
            gprime  = mu_eta(eta) # gprime = link.inverse_deriv(eta)
            z = np.array(eta + (b[k:(k+A.shape[0])] - g)/gprime)
            k=k+A.shape[0]
            
            varianceFam=family.variance
            linkinvFam=link.inverse
            g      = linkinvFam(eta)
            varg   = varianceFam(g)
            
            W=gprime**2 / varg
            W=W.reshape(len(W),1)
            cross2=np.matmul(np.transpose(A), np.asarray(W.reshape(-1)*z))
            ATWz = ATWz + cross2
            cross1=np.matmul(np.transpose(A), np.asarray(W*A))
            ATWA   = ATWA + cross1
        xold = x
        C, rank, piv = cholesky_pivot(ATWA, full_pivot=True)
        if rank<C.shape[1]:
            raise LinAlgError("Rank-deficiency detected.")
        x=solve_triangular(np.transpose(C), ATWz[piv-1], lower=True)
        x=solve_triangular(C, x, lower=False)[piv-1]
        if( np.sqrt(np.matmul(np.transpose(x-xold), x-xold)) < tol): break
    if headerNames is not None:
        x=pd.DataFrame(x, headerNames)
    elif header is not None:
        x=pd.DataFrame(x, index =list(rowA))
    else:
        x=pd.DataFrame(x)
    return (x, j)


