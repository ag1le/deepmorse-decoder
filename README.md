# deepmorse-decoder
Deep Learning Morse Decoder created by Mauri Niininen (AG1LE). 


##  Getting Started

Make sure you have Python 3.6.5 or later available in your system. 
Clone this Github repository, and create a virtual Python environment. 
Install Python libraries using requirements.txt file

```
git clone https://github.com/ag1le/deepmorse-decoder.git
python3 -m venv venv
source venv/bin/activate
pip install -r  requirements.txt
```

##  Start the program

``` 
python  specgram.py
````

You should see the program starting and a spectrogram display should pop up. 
The program is listening your microphone using pyaudio library. 
You can play an audio source with Morse code and you should now see the 4 second buffer in the spectrogram display. 

##  The CNN-LSTM-CTC model 
The model files are stored in the mymodel/ directory. 
You can create or retrain the model using the MorseDecoder.py in the morse/ directory. 
For instructions you can use the --help option. 

```
python morse/MorseDecoder.py -h 
```
