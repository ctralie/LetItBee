"""
Programmer: Chris Tralie
Purpose: Implementing the NMF techniques in [1], as well as 
plotting utilities
[1] Driedger, Jonathan, Thomas Praetzlich, and Meinard Mueller. 
"Let it Bee-Towards NMF-Inspired Audio Mosaicing." ISMIR. 2015.
"""
import numpy as np
import scipy.io as sio
import scipy.ndimage
import matplotlib.pyplot as plt
import time
import librosa
import librosa.display

def getKLError(V, WH, eps = 1e-10):
    """
    Return the Kullback-Liebler diverges between V and W*H
    """
    denom = np.array(WH)
    denom[denom == 0] = 1
    arg = V/denom
    arg[arg < eps] = eps
    return np.sum(V*np.log(arg)-V+WH)


def plotNMFSpectra(V, W, H, iter, errs, hopLength = -1):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param errs: Convergence errors
    :param hopLength: The hop length (for plotting)
    """
    plt.subplot(221)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("V")
    plt.subplot(223)
    WH = W.dot(H)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(WH), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("W*H Iteration %i"%iter)  
    plt.subplot(222)
    plt.imshow(np.log(H + np.min(H[H > 0])), cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("H Iteration %i"%iter)
    plt.subplot(224)
    errs = np.array(errs)
    errs[0] = errs[1]
    plt.semilogy(errs)
    plt.ylim([0.7*np.min(errs[errs > 0]), 1.3*np.max(errs[1::])])
    plt.title("KL Errors")
    plt.xlabel("Iteration")
    plt.tight_layout()

def plotInitialW(W, hopLength = -1):
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(W), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')        
    else:
        plt.imshow(W, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("W")

def doNMFDriedger(V, W, L, r = 7, p = 10, c = 3, plotfn = None, plotfnw = None):
    """
    Implement the technique from "Let It Bee-Towards NMF-Inspired
    Audio Mosaicing"
    :param V: M x N target matrix
    :param W: An M x K matrix of template sounds in some time order\
        along the second axis
    :param L: Number of iterations
    :param r: Width of the repeated activation filter
    :param p: Degree of polyphony; i.e. number of values in each column\
        of H which should be un-shrunken
    :param c: Half length of time-continuous activation filter
    """
    N = V.shape[1]
    K = W.shape[1]
    tic = time.time()
    H = np.random.rand(K, N)
    print("H.shape = ", H.shape)
    print("Time elapsed H initializing: %.3g"%(time.time() - tic))
    errs = np.zeros(L+1)
    errs[0] = getKLError(V, W.dot(H))
    if plotfnw:
        plt.figure(figsize=(12, 3))
        plotfnw(W)
        plt.savefig("Driedger_W.svg", bbox_inches='tight')
    if plotfn:
        res=4
        plt.figure(figsize=(res*2, res*2))
    for l in range(L):
        print("NMF Driedger iteration %i of %i"%(l+1, L))   
        iterfac = 1-float(l+1)/L       
        tic = time.time()
        #Step 1: Avoid repeated activations
        print("Doing Repeated Activations...")
        MuH = scipy.ndimage.filters.maximum_filter(H, size=(1, r))
        H[H<MuH] = H[H<MuH]*iterfac
        #Step 2: Restrict number of simultaneous activations
        print("Restricting simultaneous activations...")
        #Use partitions instead of sorting for speed
        colCutoff = -np.partition(-H, p, 0)[p, :] 
        H[H < colCutoff[None, :]] = H[H < colCutoff[None, :]]*iterfac
        #Step 3: Supporting time-continuous activations
        if c > 0:                    
            print("Supporting time-continuous activations...")
            di = K-1
            dj = 0
            for k in range(-H.shape[0]+1, H.shape[1]):
                z = np.cumsum(np.concatenate((np.zeros(c), np.diag(H, k), np.zeros(c))))
                x2 = z[2*c::] - z[0:-2*c]
                H[di+np.arange(len(x2)), dj+np.arange(len(x2))] = x2
                if di == 0:
                    dj += 1
                else:
                    di -= 1
        #KL Divergence Version
        WH = W.dot(H)
        WH[WH == 0] = 1
        VLam = V/WH
        WDenom = np.sum(W, 0)
        WDenom[WDenom == 0] = 1
        H = H*((W.T).dot(VLam)/WDenom[:, None])
        print("Elapsed Time H Update %.3g"%(time.time() - tic))
        errs[l+1] = getKLError(V, W.dot(H))
        #Output plots every 20 iterations
        if plotfn and ((l+1)==L or (l+1)%20 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMFDriedger_%i.png"%(l+1), bbox_inches = 'tight')
    return H
