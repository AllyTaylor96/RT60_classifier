import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import argparse

# load in file location
parser = argparse.ArgumentParser(description='Test RT60',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('test_path', help='Destination of test files')
args = parser.parse_args()

data_direc = Path(args.test_path)
# send to cuda line
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load in classifier classes, saved model and put into eval mode
classifier_classes = pickle.load(open('results/classes', 'rb'))
model = torch.load('rt60_classifier.pt')
model.eval()
if torch.cuda.is_available():
        model.cuda()

# iterate through files in directory
results = {}
results_00 = []
results_02 = []
results_04 = []
results_06 = []
results_08 = []
results_10 = []

for path in sorted(data_direc.glob('*.pt')):
    spec = torch.load(path).unsqueeze(0)
    spec = spec.to(device)
    prob = F.softmax(model(spec), dim=1)
    pred_class = classifier_classes[int(prob.argmax())]
    results[str(path.stem)] = pred_class
    if str(path.stem).startswith('rt0.0'):
        results_00.append(pred_class)
    elif str(path.stem).startswith('rt0.2'):
        results_02.append(pred_class)
    elif str(path.stem).startswith('rt0.4'):
        results_04.append(pred_class)
    elif str(path.stem).startswith('rt0.6'):
        results_06.append(pred_class)
    elif str(path.stem).startswith('rt0.8'):
        results_08.append(pred_class)
    elif str(path.stem).startswith('rt1.0'):
        results_10.append(pred_class)

# print amount of each class detected
#from pprint import pprint
#pprint(results)
with open('results/test_results.txt', 'w') as f:
    f.write('Actual RT: 0.0 | Predictions: {} \n'.format(Counter(results_00)))
    f.write('Actual RT: 0.2 | Predictions: {} \n'.format(Counter(results_02)))
    f.write('Actual RT: 0.4 | Predictions: {} \n'.format(Counter(results_04)))
    f.write('Actual RT: 0.6 | Predictions: {} \n'.format(Counter(results_06)))
    f.write('Actual RT: 0.8 | Predictions: {} \n'.format(Counter(results_08)))
    f.write('Actual RT: 1.0 | Predictions: {}'.format(Counter(results_10)))


