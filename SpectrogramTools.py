import numpy as np
import pyrubberband as pyrb

halfsine = lambda W: np.sin(np.pi*np.arange(W)/float(W))

def STFT(X, W, H, winfunc = None, useLibrosa = True):
    """
    :param X: An Nx1 audio signal
    :param W: A window size
    :param H: A hopSize
    :param winfunc: Handle to a window function
    """
    if useLibrosa:
        import librosa
        return librosa.core.stft(X, n_fft=W, hop_length=H, window = 'blackman')
    Q = W/H
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    if not winfunc:
        #Use half sine by default
        winfunc = halfsine
    win = winfunc(W)
    NWin = int(np.floor((X.size - W)/float(H)) + 1)
    S = np.zeros((W, NWin), dtype = complex)
    for i in range(NWin):
        S[:, i] = np.fft.fft(win*X[np.arange(W) + (i-1)*H])
    #Second half of the spectrum is redundant for real signals
    if W%2 == 0:
        #Even Case
        S = S[0:int(W/2)+1, :]
    else:
        #Odd Case
        S = S[0:int((W-1)/2)+1, :]
    return S

def iSTFT(pS, W, H, winfunc = None, useLibrosa = True):
    """
    :param pS: An NBins x NWindows spectrogram
    :param W: A window size
    :param H: A hopSize
    :param winfunc: Handle to a window function
    :returns S: Spectrogram
    """
    if useLibrosa:
        import librosa
        return librosa.core.istft(pS, hop_length = H, window = 'blackman')
    #First put back the entire redundant STFT
    S = np.array(pS, dtype = complex)
    if W%2 == 0:
        #Even Case
        S = np.concatenate((S, np.flipud(np.conj(S[1:-1, :]))), 0)
    else:
        #Odd Case
        S = np.concatenate((S, np.flipud(np.conj(S[1::, :]))), 0)
    
    #Figure out how long the reconstructed signal actually is
    N = W + H*(S.shape[1] - 1)
    X = np.zeros(N, dtype = complex)
    
    #Setup the window
    Q = W/H;
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    if not winfunc:
        #Use half sine by default
        winfunc = halfsine
    win = winfunc(W)
    win = win/(Q/2.0)

    #Do overlap/add synthesis
    for i in range(S.shape[1]):
        X[i*H:i*H+W] += win*np.fft.ifft(S[:, i])
    return X

def getPitchShiftedSpecs(X, Fs, W, H, shiftrange = 6, GapWins = 10):
    """
    Concatenate a bunch of pitch shifted versions of the spectrograms
    of a sound, using the rubberband library
    :param X: A mono audio array
    :param Fs: Sample rate
    :param W: Window size
    :param H: Hop size
    :param shiftrange: The number of halfsteps below and above which \
        to shift the sound
    :returns SRet: The concatenate spectrogram
    """
    SRet = np.array([])
    for shift in range(-shiftrange, shiftrange+1):
        print("Computing STFT pitch shift %i"%shift)
        if shift == 0:
            Y = np.array(X)
        else:
            Y = pyrb.pitch_shift(X, Fs, shift)
        S = STFT(Y, W, H)
        Gap = np.zeros((S.shape[0], GapWins), dtype=complex)
        if SRet.size == 0:
            SRet = S
        else:
            SRet = np.concatenate((SRet, Gap, S), 1)
    return SRet

def griffinLimInverse(S, W, H, NIters = 10, winfunc = None):
    """
    Do Griffin Lim phase retrieval
    :param S: An NFreqsxNWindows spectrogram
    :param W: Window size used in STFT
    :param H: Hop length used in STFT
    :param NIters: Number of iterations to go through (default 10)
    :winfunc: A handle to a window function (None by default, use halfsine)
    :returns: An Nx1 real signal corresponding to phase retrieval
    """
    eps = 2.2204e-16
    if not winfunc:
        winfunc = halfsine
    A = np.array(S, dtype = complex)
    for i in range(NIters):
        print("Iteration %i of %i"%(i+1, NIters))
        A = STFT(iSTFT(A, W, H, winfunc), W, H, winfunc)
        Norm = np.sqrt(A*np.conj(A))
        Norm[Norm < eps] = 1
        A = np.abs(S)*(A/Norm)
    X = iSTFT(A, W, H, winfunc)
    return np.real(X)

def getPitchShiftedSpecsFromSpec(S, Fs, W, H, shiftrange = 6, GapWins = 10):
    """
    Same as getPitchShiftedSpecs, except input a spectrogram, which needs
    to be inverted, instead of a sound
    """
    X = griffinLimInverse(S, W, H)
    return getPitchShiftedSpecs(X, Fs, W, H, shiftrange, GapWins)

