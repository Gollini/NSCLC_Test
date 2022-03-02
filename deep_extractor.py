import argparse
import pandas as pd
import torch
from torchvision.models._utils import IntermediateLayerGetter
from utils.models import DirectPredictionCT
from utils.dataset import Dataset

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='./data')
argparser.add_argument('--exp_model', type=str, default='resnet18')
argparser.add_argument('--image_channels', type=int, default=3, help='options: 3, 9')
argparser.add_argument('--image_size', type=int, default=224, help='options: 32, 64, 112, 224')
argparser.add_argument('--load_manual_features', action='store_true')
argparser.add_argument('--transforms', type=str, default=None)

param = argparser.parse_args()

print('--------------------------------')
# GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA Available')
else:
    device = torch.device("cpu")
    print('GPU Not available')
print('--------------------------------')

dataset = Dataset(param)

y = dataset.labels

dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=len(y))

ct_model = DirectPredictionCT(param)
state_dict = torch.load('/home/ivo.navarrete/Desktop/NSCLC/NSCLC_Test/checkpoints/ig_105/CT-model-fold-0.pth')#['state_dict']
# print(state_dict.keys())
# new_dict = {k.replace('model.',''): state_dict[k] for k in list(state_dict.keys())}
# print(new_dict.keys())
ct_model.load_state_dict(state_dict)

ct_model = ct_model.model
extractor = IntermediateLayerGetter(ct_model, {'avgpool':'out'}).to(device)


# print(extractor)

for i, data in enumerate(dataloader, 0):

    # Get inputs
    x_frames, labels = data
    x_frames= x_frames.float().to(device)

    #foward pass
    deep_features = extractor(x_frames)['out'].squeeze()

deep_features = deep_features.cpu().detach().numpy()

df = pd.DataFrame(deep_features, index=y.index)
df.to_csv('/home/ivo.navarrete/Desktop/NSCLC/NSCLC_Test/data/deep_data.csv')

print('Deep Extraction performed')