import torch
import argparse
import numpy as np
import json

from model import PedalNet

def convert(args):
    ''' Converts a *.ckpt model from PedalNet into a .json format used in WaveNetVA. 
        TODO: Based on the high values of the linear_mix layer weights, there are still unaccounted 
              differences between the Pytorch and Tensorflow implementations. Need to understand 
              before successfully converting a model for WaveNetVA. (linear mix weights ranged from 20 to 60,
              and caused automute in DAW because the levels produced were too high. After manually reducing 
              the linear mix weights a factor of 3, was able to play in DAW, but model was not correct
              (some musical distortion of ts9 model, but not anywhere close to PedalNet converted .wav files of same model)

              Note: After stopping the model training at various epochs, it appears the linear_mix layer weight values
                    are growing more or less linearly with each epoch, starting around .5 @epoch=10, ~25 @epoch=750, ~50 @epoch=1500
                    The predicted wave files sound like the target, so this is probably correct for this Pytorch model, but not 
                    sure what is different from the WaveNetVA model.

              Note: in WaveNetVa, the b_out values for the last layer of all trained models is 0.0, not sure why, and this 
                    is not the case for this PedalNet model. 
              
              Current changes to the PedalNet model to match WaveNetVA include:
                1. Added CausalConv1d() to use causal padding
                2. Added an input layer, which is a Conv1d(in_channls=1, out_channels=num_channels, kernel_size=1)
              
              The model parameters used for conversion testing match the Wavenetva1 model (untested with other parameters):
              --num_channels=16, --dilation_depth=10, --num_repeat=1, --kernel_size=3
    '''

    # Permute tensors to match Tensorflow format with .permute(a,b,c):
    #a, b, c = 0, 1, 2  # Original shape
    a, b, c = 2, 1, 0  # Pytorch uses (out_channels, in_channels, kernel_size), TensorFlow uses (kernel_size, in_channels, out_channels)

    model = PedalNet.load_from_checkpoint(checkpoint_path=args.model)

    sd = model.state_dict()

    # Get hparams from model
    hparams = model.hparams
    residual_channels = hparams["num_channels"]
    filter_width = hparams["kernel_size"]
    dilations = [2 ** d for d in range(hparams["dilation_depth"])] * hparams["num_repeat"]

    data_out = {"activation": "gated", 
                "output_channels": 1, 
                "input_channels": 1, 
                "residual_channels": residual_channels, 
                "filter_width": filter_width, 
                "dilations": dilations, 
                "variables": []}

    # Use pytorch model data to populate the json data for each layer
    for i in range(-1, len(dilations) + 1):
        # Input Layer
        if i == -1: 
            data_out["variables"].append({"layer_idx":i,
                                        "data":[str(i) for i in (sd['wavenet.input_layer.weight']).permute(a,b,c).flatten().numpy().tolist()],
                                        "name":"W"})
            data_out["variables"].append({"layer_idx":i,
                                        "data":[str(i) for i in (sd['wavenet.input_layer.bias']).flatten().numpy().tolist()],
                                        "name":"b"})
        # # Linear Mix Layer
        #KAB Note: Linear mix weight/bias seemed large (20 to 60 range) and caused automute in DAW (too loud) 
        #          Tested scaling down the linear mix weight and bias by /1000, worked in DAW but not correct model 
        # TODO: Figure out why the values are so large compared to WaveNetVA, linear_mix layer differences?
        #       Note: The final hidden layer biases are always all "0.0" in the WaveNetVA models, why?

        elif  i == len(dilations):  
            data_out["variables"].append({"layer_idx":i,
                                        "data":[str(i) for i in (sd['wavenet.linear_mix.weight']).permute(a,b,c).flatten().numpy().tolist()], 
                                        "name":"W"})

            data_out["variables"].append({"layer_idx":i,
                                        "data":[str(i) for i in (sd['wavenet.linear_mix.bias']).numpy().tolist()],
                                        "name":"b"})
        # Hidden Layers
        else:
            data_out["variables"].append({"layer_idx":i,
                                    "data":[str(i) for i in sd['wavenet.convs_tanh.' + str(i) + '.weight'].permute(a,b,c).flatten().numpy().tolist() +
                                    sd['wavenet.convs_sigm.' + str(i) + '.weight'].permute(a,b,c).flatten().numpy().tolist()], 
                                    "name":"W_conv"})
            data_out["variables"].append({"layer_idx":i,
                                        "data":[str(i) for i in sd['wavenet.convs_tanh.' + str(i) + '.bias'].flatten().numpy().tolist() + 
                                        sd['wavenet.convs_sigm.' + str(i) + '.bias'].flatten().numpy().tolist()],
                                        "name":"b_conv"})
            data_out["variables"].append({"layer_idx":i,
                                        "data":[str(i) for i in sd['wavenet.residuals.' + str(i) + '.weight'].permute(a,b,c).flatten().numpy().tolist()],
                                        "name":"W_out"})
            data_out["variables"].append({"layer_idx":i,
                                        "data":[str(i) for i in sd['wavenet.residuals.' + str(i) + '.bias'].flatten().numpy().tolist()],
                                        "name":"b_out"})

    #for debugging ###################
    print("State Dict Data:")
    for i in sd.keys():
        print(i, "  Shape: ", sd[i].shape)

    # output final dictionary to json file
    with open('converted_model.json', 'w') as outfile:
        json.dump(data_out, outfile)
    print("Need to remove the quotations around number values, can use  https://csvjson.com/json_beautifier")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pedalnet.ckpt")
    args = parser.parse_args()
    convert(args)