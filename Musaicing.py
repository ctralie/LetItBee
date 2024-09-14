"""
Programmer: Chris Tralie
Purpose: To serve as an entry point for Driedger's Musaicing Technique
"""
import numpy as np
import scipy.io as sio
import argparse
import librosa
from SpectrogramTools import *
from NMF import *

def doMusaicing(source, target, result, sr = 22050, winSize = 2048, hopSize = 1024, \
                    NIters = 80, r = 7, p = 10, c = 3, savePlots = True, \
                    pitchShift = 0, plotEvery = 20):
    """
    :param source: Source audio filename
    :param target: Target audio filename
    :param result: Result wavfile path
    :param winSize: Window Size of STFT in samples
    :param hopSize: Hop size of STFT in samples
    :param NIters: Number of iterations of Driedger's technique
    :param r: Width of the repeated activation filter
    :param p: Degree of polyphony; i.e. number of values in each column\
        of H which should be un-shrunken
    :param c: Half length of time-continuous activation filter
    :param savePlots: Whether to save plots showing progress of NMF \
        every 20 iterations
    :param pitchShift: How many halfsteps to pitch shift the corpus up and down
    :param plotEvery: Save plot to disk every time this many iterations passes
    """
    X, sr = librosa.load(source, sr=sr)
    WComplex = getPitchShiftedSpecs(X, sr, winSize, hopSize, pitchShift)
    W = np.abs(WComplex)
    X, sr = librosa.load(target, sr=sr)
    V = np.abs(STFT(X, winSize, hopSize))
    fn = None
    fnw = None
    if savePlots:
        fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
        fnw = lambda W: plotInitialW(W, hopSize)
    H = doNMFDriedger(V, W, NIters, r=r, p=p, c=c, plotfn=fn, plotfnw = fnw, plotEvery = plotEvery)
    H = np.array(H, dtype=complex)
    V2 = WComplex.dot(H)
    print("Doing phase retrieval...")
    Y = griffinLimInverse(V2, winSize, hopSize, NIters=30)
    Y = Y/np.max(np.abs(Y))
    sio.wavfile.write(result, sr, Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help="Path to audio file for source sounds")
    parser.add_argument('--target', type=str, required=True, help="Path to audio file for target sound")
    parser.add_argument('--result', type=str, required=True, help="Path to wav file to which to save the result")
    parser.add_argument('--sr', type=int, default=22050, help="Sample rate")
    parser.add_argument('--winSize', type=int, default=2048, help="Window Size in samples")
    parser.add_argument('--hopSize', type=int, default=512, help="Hop Size in samples")
    parser.add_argument('--NIters', type=int, default=60, help="Number of iterations of NMF")
    parser.add_argument('--r', type=int, default=7, help="Width of the repeated activation filter")
    parser.add_argument('--p', type=int, default=10, help="Degree of polyphony; i.e. number of values in each column of H which should be un-shrunken")
    parser.add_argument('--c', type=int, default=3, help="Half length of time-continuous activation filter")
    parser.add_argument('--saveplots', type=int, default=0, help='Save plots of iterations to disk')
    parser.add_argument('--pitchShift', type=int, default=0, help="Amount of halfsteps to pitch shift the corpus up and down")
    parser.add_argument('--plotEvery', type=int, default=20, help="If plotting, make a plot after every interval of this many iterations")
    opt = parser.parse_args()
    doMusaicing(opt.source, opt.target, opt.result, sr=opt.sr, winSize=opt.winSize, \
                hopSize=opt.hopSize, NIters=opt.NIters, r=opt.r, p=opt.p, c=opt.c, \
                savePlots=opt.saveplots, pitchShift=opt.pitchShift, plotEvery=opt.plotEvery)