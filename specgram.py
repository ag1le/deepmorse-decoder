"""
Created by Mauri Niininen (AG1LE)
Real time Morse decoder using CNN-LSTM-CTC Tensorflow model

adapted from https://github.com/ayared/Live-Specgram

"""
############### Import Libraries ###############
from matplotlib.mlab import specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2


############### Import Modules ###############
import mic_read
from morse.MorseDecoder import  Config, Model, Batch, DecoderType


############### Constants ###############
SAMPLES_PER_FRAME = 4 #Number of mic reads concatenated within a single window
nfft = 256 # NFFT value for spectrogram
overlap = nfft-56 # overlap value for spectrogram
rate = mic_read.RATE #sampling rate


############### Call Morse decoder ###############
def infer_image(model, img):
    if img.shape == (128, 32):
        batch = Batch(None, [img])
        (recognized, probability) = model.inferBatch(batch, True)
        return img, recognized, probability
    else:
        print(f"ERROR: img shape:{img.shape}")

# Load the Tensorlow model 
config = Config('model.yaml')
model = Model(open("morseCharList.txt").read(), config, decoderType = DecoderType.BestPath, mustRestore=True)

stream,pa = mic_read.open_mic()


############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""
def get_sample(stream,pa):
    data = mic_read.get_data(stream,pa)
    return data
"""
get_specgram:
takes the FFT to create a spectrogram of the given audio signal
input: audio signal, sampling rate
output: 2D Spectrogram Array, Frequency Array, Bin Array
see matplotlib.mlab.specgram documentation for help
"""
def get_specgram(signal,rate):
    arr2D,freqs,bins = specgram(signal,window=np.blackman(nfft),  
                                Fs=rate, NFFT=nfft, noverlap=overlap,
                                pad_to=32*nfft   )
    return arr2D,freqs,bins

"""
update_fig:
updates the image, just adds on samples at the start until the maximum size is
reached, at which point it 'scrolls' horizontally by determining how much of the
data needs to stay, shifting it left, and appending the new data. 
inputs: iteration number
outputs: updated image
"""
def update_fig(n):
    data = get_sample(stream,pa)
    arr2D,freqs,bins = get_specgram(data,rate)
    
    im_data = im.get_array()
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data,np.s_[:-keep_block],1)
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)

    # Get the image data array shape (Freq bins, Time Steps)
    shape = im_data.shape

    # Find the CW spectrum peak - look across all time steps
    f = int(np.argmax(im_data[:])/shape[1])

    # Create a 32x128 array centered to spectrum peak 
    if f > 16: 
        print(f"n:{n} f:{f}")
        img = cv2.resize(im_data[f-16:f+16][0:128], (128,32)) 
        if img.shape == (32,128):
            cv2.imwrite("dummy.png",img)
            img = cv2.transpose(img)
            img, recognized, probability = infer_image(model, img)
            if probability > 0.0000001:
                print(f"infer_image:{recognized} prob:{probability}")
    return im,

def main():
    
    global im
    ############### Initialize Plot ###############
    fig = plt.figure()
    """
    Launch the stream and the original spectrogram
    """
    stream,pa = mic_read.open_mic()
    data = get_sample(stream,pa)
    arr2D,freqs,bins = get_specgram(data,rate)
    """
    Setup the plot paramters
    """
    extent = (bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])
    
    im = plt.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                    cmap = 'Greys',norm = None) 

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Real Time Spectogram')
    plt.gca().invert_yaxis()
    #plt.colorbar() #enable if you want to display a color bar

    ############### Animate ###############
    anim = animation.FuncAnimation(fig,update_fig,blit = True,
                                interval=mic_read.CHUNK_SIZE/1000)

                                
    try:
        plt.show()
    except:
        print("Plot Closed")

    ############### Terminate ###############
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("Program Terminated")

if __name__ == "__main__":
    main()