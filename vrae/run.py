from vrae import VRAE
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset


parser = argparse.ArgumentParser(description='Variational Recurrent AutoEncoder')
parser.add_argument('-hs', '--hidden-size', help = 'Hidden size of RNN', default = 90, type = int, required = False)
parser.add_argument('-hd', '--hidden-layer-depth', help = 'Hidden layer depth of RNN', default = 2, type = int, required = False)
parser.add_argument('-ll', '--latent-length', help = 'Length of the latent vector', default = 20, type = int, required = False)
parser.add_argument('-bs', '--batch-size', help = 'Batch size per epoch', default = 32, type = int, required = False)
parser.add_argument('-lr', '--learning-rate', help = 'Learning rate', default = 0.005, type = float, required = False)
parser.add_argument('-n', '--n-epochs', help = 'Number of epochs', default = 5, type = int, required = False)
parser.add_argument('-d', '--dropout-rate', help = 'Encoder dropout rate', type = float, default = 0.2,required = False)
parser.add_argument('-o', '--optimizer', help = 'Optimizer to be used', default = 'Adam',required = False)
parser.add_argument('-g', '--cuda', help = 'Boolean, GPU to be used or not', default = False, type = bool, required = False)
parser.add_argument('-p', '--print-every', help = 'Print output every p iterations', default = 100,type = int, required = False)
parser.add_argument('-c', '--clip', help = 'Clips the gradients, specified by max_grad_norm', type = int, default = True,required = False)
parser.add_argument('-m', '--max-grad-norm', help = 'Amount to be clipped by', default = 5, type = int, required = False)
parser.add_argument('-lo', '--loss', help = 'Loss function to be used', default = 'ReconLoss',required = False)
parser.add_argument('-b', '--block', help = 'Basic building block of encoder/decoder', default = 'LSTM',required = False)
parser.add_argument('-dl', '--dload', help = 'Download directory', default = '.', required = False)

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

np_data = np.genfromtxt('../data/ECG5000.csv', delimiter=',')
dataset = TensorDataset(torch.from_numpy(np_data[:,1:]))

sequence_length = dataset.tensors[0].shape[1]
vrae = VRAE(sequence_length=sequence_length, **args_dict)

vrae.fit_transform(dataset, save = True)