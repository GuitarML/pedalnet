import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy import signal
import argparse

def analyze_pred_vs_actual(args):
    ''' Generate plots to analyze the predicted signal vs the actual
        signal. 
        
        Inputs: 
            file1 : The actual signal, by default will use y_test.wav from the test.py output
            file2 : The predicted signal, by default will use y_pred.wav from the test.py output
            model_name : Used to add the model name to the plot .png filename
            show_plots : Default is 1 to show plots, 0 to only generate .png files and suppress plots

        1. Plots the two signals
        2. Calculates Error to signal ratio (for overall model evaluation):
           sum(pred_signal - actual_signal)) / sum(actual_signal)  
        3. Plots the absolute value of pred_signal - actual_signal  (to visualize error over time)
        4. Plots the spectrogram of (pred_signal - actual signal) 
             The idea here is to show problem frequencies from the model training
    '''
    file1 = args.file1
    file2 = args.file2
    model_name = args.model_name
    show_plots = args.show_plots

    # Extract Audio from Wav File1
    spf = wave.open(file1, "r")
    signal1 = spf.readframes(-1)
    signal1 = np.fromstring(signal1, "Int16")
    fs = spf.getframerate()

    # Ensure mono
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    Time = np.linspace(0, len(signal1) / fs, num=len(signal1))

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(13, 8))
    fig.suptitle('Predicted vs Actual Signal')
    ax1.plot(Time, signal1, label=file1, color='red')

    # Extract Audio from Wav File2
    spf2 = wave.open(file2, "r")
    signal2 = spf2.readframes(-1)
    signal2 = np.fromstring(signal2, "Int16")
    fs2 = spf2.getframerate()

    # Ensure mono
    if spf2.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    Time2 = np.linspace(0, len(signal2) / fs2, num=len(signal2))
    ax1.plot(Time2, signal2, label=file2, color='green')
    ax1.legend(loc='upper right')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Wav File Comparison")
    ax1.grid('on')

    # Calculate error to signal ratio
    error_list = []
    for s1, s2 in zip(signal1, signal2):
        error_list.append(abs(s2 - s1))

    all_positive_signal = []
    for s3 in signal1:
        all_positive_signal.append(float(abs(s3)))
    error2 = abs(sum(error_list) / sum(all_positive_signal))
    print("Error to Signal Ratio: ", error2*100, "%")

    # Plot signal difference
    signal_diff = signal2 - signal1
    ax2.plot(Time2, error_list, label="signal diff", color='blue')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("abs(pred_signal-actual_signal) Error to Signal Ratio: "+str(round(error2*100,4))+"%")
    ax2.grid('on')
    plt.savefig(model_name+'_signal_comparison_err_'+str(round(error2*100,2))+'.png',bbox_inches='tight')

    # Plot spectrogram difference
    plt.figure(figsize=(12, 8))
    print("Creating spectrogram data..")
    frequencies, times, spectrogram = signal.spectrogram(signal_diff, 44100)
    plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram))
    plt.colorbar()
    plt.title("Diff Spectrogram")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(model_name+'_diff_spectrogram.png', bbox_inches='tight')

    if show_plots == 1:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", default="y_test.wav")
    parser.add_argument("--file2", default="y_pred.wav")
    parser.add_argument("--model_name", default='plot')
    parser.add_argument("--show_plots", default=1)
    args = parser.parse_args()
    analyze_pred_vs_actual(args)