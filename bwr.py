import numpy as np
import pywt

def calc_baseline(signal):
    """
    Calculate the baseline of signal.

    Args:
        signal (numpy 1d array): signal whose baseline should be calculated


    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))
    
    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        
        #The lead contains too much noise to remove the baseline wander
        if iterations > 100:
            baseline = np.zeros((len(signal)))
            return baseline
        
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]


def remove_baseline_wander_wavelet(ecg_signal):
    # Choose a wavelet (e.g., db4)
    wavelet = 'db4'

    # Decompose the signal to get approximation (cA1) and detail (cD1) coefficients
    cA1, cD1 = pywt.dwt(ecg_signal, wavelet)

    # Reconstruct the signal without baseline wander
    ecg_without_baseline = pywt.idwt(cA1, None, wavelet)

    return ecg_without_baseline
