import numpy as np
import logging
from ridge import ridge, ridge_corr, bootstrap_ridge

logging.basicConfig(level=logging.DEBUG)

# Create some test data
N = 200 # features
M = 1000 # response sources (voxels, whatever)
TR = 1000 # regression timepoints
TP = 200 # prediction timepoints

snrs = np.linspace(0, 0.2, M)
realwt = np.random.randn(N, M)
features = np.random.randn(TR+TP, N)
realresponses = np.dot(features, realwt) # shape (TR+TP, M)
noise = np.random.randn(TR+TP, M)
responses = (realresponses * snrs) + noise

Rresp = responses[:TR]
Presp = responses[TR:]
Rstim = features[:TR]
Pstim = features[TR:]

# Run bootstrap ridge
wt, corr, valphas, bscorrs, valinds = bootstrap_ridge(Rstim, Rresp, Pstim, Presp,
                                                      alphas=np.logspace(-2, 2, 20),
                                                      nboots=5,
                                                      chunklen=10, nchunks=15)


# Corr should increase quickly across "voxels". Last corr should be large (>0.9-ish).
# wt should be very similar to realwt for last few voxels.
