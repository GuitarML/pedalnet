---------------------------------------------------------------------------
This modified version of PedalNet is meant to be used in 
conjuction with the WaveNetVA code repository. You can train a model using 
this repo, then convert it to a .json model that can be loaded into the 
WaveNetVA plugin. 

Usage:

	python convert_pedalnet_to_wavnetva.py --model=your_trained_model.ckpt

Generates a file named "converted_model.json" that can be loaded into the
WaveNetVa plugin.

You can also use "plot_wav.py" to evaluate the trained PedalNet model. By 
default, this will analyze the three .wav files from the test.py output. It 
will output analysis plots and calculate the error to signal ratio. 

Usage (after running "python test.py --model=your_model.ckpt"):

	python plot_wav.py

Note: The training wav files in data/ are float32 format, as opposed to int16,
and the scripts in this repo are modified to use float32.

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

