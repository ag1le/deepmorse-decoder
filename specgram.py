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
import sys 
from matplotlib.widgets import TextBox
from fuzzysearch import find_near_matches


############### Import Modules ###############
import mic_read
from morse.MorseDecoder import Config, Model, Batch, DecoderType


############### Constants ###############
SAMPLES_PER_FRAME = 4  # Number of mic reads concatenated within a single window
nfft = 256  # NFFT value for spectrogram
overlap = nfft - 56  # overlap value for spectrogram
rate = mic_read.RATE  # sampling rate


############### Call Morse decoder ###############
def infer_image(model, img):
    if img.shape == (128, 32):
        try:
            batch = Batch(None, [img])
            (recognized, probability) = model.inferBatch(batch, True)
            return img, recognized, probability
        except Exception as err:
            print(f"ERROR:{err}")
    else:
        print(f"ERROR: img shape:{img.shape}")
    return '', '', 0.0





############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""


def get_sample(stream, pa):
    data = mic_read.get_data(stream, pa)
    return data


"""
get_specgram:
takes the FFT to create a spectrogram of the given audio signal
input: audio signal, sampling rate
output: 2D Spectrogram Array, Frequency Array, Bin Array
see matplotlib.mlab.specgram documentation for help
"""


def get_specgram(signal, rate):
    arr2D, freqs, bins = specgram(
        signal,
        window=np.blackman(nfft),
        Fs=rate,
        NFFT=nfft,
        noverlap=overlap,
        pad_to=32 * nfft,
    )
    return arr2D, freqs, bins


class TextBuffer():
    """Buffer to display decoded text """

    def __init__(self, length):
        self.buffer = '*'*length
        self.length = length

    def update_text(self, string):
        """ scrolling text buffer with matching logic"""
        try:
            matches = find_near_matches(string, self.buffer, max_l_dist=3)
        except:
            matches = None
        print(f"string:{string}")
        if matches:
            print(f"{self.buffer}")
            for match in matches:
                print(f"{' ': <{match.start}}{match.matched}")
            if match.start + len(match.matched) < self.length:  # match found but not at the end, just append
                mybuf = self.buffer[len(string):self.length] + string
            else:                                               # math found - append string and scroll 
                mybuf = self.buffer[len(string)-len(match.matched):match.start] + string
        
        else:                                                   # no match, just append the string 
            mybuf = self.buffer[len(string):self.length] + string
        self.buffer = mybuf[0:self.length]
        return self.buffer

global buffer
buffer = TextBuffer(40)


"""
update_fig:
updates the image, just adds on samples at the start until the maximum size is
reached, at which point it 'scrolls' horizontally by determining how much of the
data needs to stay, shifting it left, and appending the new data. 
inputs: iteration number
outputs: updated image
"""
def update_fig(n, text_box):

    data = get_sample(stream, pa)
    arr2D, freqs, bins = get_specgram(data, rate)

    im_data = im.get_array()
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1] * (SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data, np.s_[:-keep_block], 1)
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)

    # Get the image data array shape (Freq bins, Time Steps)
    shape = im_data.shape

    # Find the CW spectrum peak - look across all time steps
    f = int(np.argmax(im_data[:]) / shape[1])

    # Create a 32x128 array centered to spectrum peak
    if f > 16:
        img = cv2.resize(im_data[f - 16 : f + 16][0:128], (128, 32))
        if img.shape == (32, 128):
            cv2.imwrite("dummy.png", img)

            # normalize
            (m, s) = cv2.meanStdDev(img)
            m = m[0][0]
            s = s[0][0]
            img = img - m
            img = img / s if s>0 else img

            img = cv2.transpose(img)
            img, recognized, probability = infer_image(model, img)
            if probability > 0.0000001:
                # Output decoded text 
                txt = buffer.update_text(f"{str(recognized[0][1:-1])}")
                text_box.set_val(txt)
                print(f"f:{f} n:{n} {txt}")
    return  im, text_box, 



def main():


    global im
    global stream
    global pa
    global model
    global fig
    ############### Initialize Plot ###############
    


    # Load the Tensorlow model
    config = Config("model_arrl3.yaml")
    model = Model(
        open(config.value("experiment.fnCharList")).read(),
        config,
        decoderType=DecoderType.BeamSearch, #,DecoderType.BestPath
        mustRestore=True,
    )

    """
    Launch the stream and the original spectrogram
    """
    stream, pa = mic_read.open_mic()
    data = get_sample(stream, pa)
    arr2D, freqs, bins = get_specgram(data, rate)
  
    """
    Setup the spectrogram plot and textbox for the decoder
    """
    fig, (axbox,ax) = plt.subplots(2,1)
    text_box = TextBox(axbox, "Morse:")
    extent = (bins[0], bins[-1] * SAMPLES_PER_FRAME, freqs[-1], freqs[0])

    im = ax.imshow(
        arr2D,
        aspect="auto",
        extent=extent,
        interpolation="none",
        cmap="Greys",
        norm=None,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Real Time Spectogram")
    plt.gca().invert_yaxis()
    # plt.colorbar() #enable if you want to display a color bar

    ############### Animate ###############
    anim = animation.FuncAnimation(
        fig, update_fig,  
        blit=False, 
        interval=mic_read.CHUNK_SIZE / 1000,
        fargs=(text_box,)
    )



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
