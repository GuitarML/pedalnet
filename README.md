---------------------------------------------------------------------------
This modified version of PedalNet is meant to be used in 
conjuction with the SmartGuitarPedal, SmartGuitarAmp, and WaveNetVA code
repositories. You can train a model using this repo, then convert it 
to a .json model that can be loaded into the VST plugin. 

The following repositories are compatible with the converted .json model,
for use with real time guitar playing through a DAW plugin or stand alone app:

https://github.com/keyth72/SmartGuitarPedal

https://github.com/keyth72/SmartGuitarAmp

https://github.com/damskaggep/WaveNetVA


Usage:

	python convert_pedalnet_to_wavnetva.py --model=your_trained_model.ckpt

Generates a file named "converted_model.json" that can be loaded into the VST plugin.

You can also use "plot_wav.py" to evaluate the trained PedalNet model. By 
default, this will analyze the three .wav files from the test.py output. It 
will output analysis plots and calculate the error to signal ratio. 

Usage (after running "python test.py --model=your_model.ckpt"):

	python plot_wav.py


Differences from the original PedalNet (to make compatible with WaveNet plugin):
1. Uses a custom Causal Padding mode not available in PyTorch.
2. Uses a single conv1d layer for both sigm and tanh calculations, instead of 
   two separate layers.
3. Requires float32 .wav files for training (instead of int16).

Helpful tips on training models:
1. Wav files should be 3 - 4 minutes long, and contain a variety of
   chords, individual notes, and playing techniques to get a full spectrum
   of data for the model to "learn" from.
2. A buffer splitter was used with pedals to obtain a pure guitar signal
   and post effect signal.
3. Obtaining sample data from an amp can be done by splitting off the original 
   signal, with the post amp signal coming from a microphone (I used a SM57).
   Keep in mind that this captures the dynamic response of the mic and cabinet.
   In the original research the sound was captured directly from within the amp
   circuit to have a "pure" amp signal.
4. Generally speaking, the more distorted the effect/amp, the more difficult it
   is to train. Experiment with different hyperparameters for each target 
   hardeware. I found that a model with only 5 channels was able to sufficiently
   model some effects, and this reduces the model size and allows the plugin 
   to use less processing power.

---------------------------------------------------------------------------

# PedalNet

Re-creation of model from [Real-Time Guitar Amplifier Emulation with Deep
Learning](https://www.mdpi.com/2076-3417/10/3/766/htm)

See my [blog
post](http://teddykoker.com/2020/05/deep-learning-for-guitar-effect-emulation/)
for a more in depth description along with song demos.

## Data

`data/in.wav` - Concatenation of a few samples from the
[IDMT-SMT-Guitar](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/guitar.html) dataset<br>
`data/ts9_out.wav` - Recorded output of `in.wav` after being passed through an
Ibanez TS9 Tube Screamer (all knobs at 12 o'clock).<br>
`models/pedalnet.ckpt` - Pretrained model weights


## Usage

**Run effect on .wav file**:
Must be single channel, 44.1 kHz
```bash
# must be same data used to train
python prepare_data.py data/in.wav data/out_ts9.wav 

# specify input file and desired output file
python predict.py my_input_guitar.wav my_output.wav 

# if you trained you own model you can pass --model flag
# with path to .ckpt
```

**Train**:
```bash
python prepare_data.py data/in.wav data/out_ts9.wav # or use your own!
python train.py 
python train.py --gpus "0,1"  # for multiple gpus
python train.py -h  # help (see for other hyperparameters)
```

**Test**:
```bash
python test.py # test pretrained model
python test.py --model lightning_logs/version_{X}/epoch={EPOCH}.ckpt  # test trained model
```
Creates files `y_test.wav`, `y_pred.wav`, and `x_test.wav`, for the ground truth
output, predicted output, and input signal respectively.

