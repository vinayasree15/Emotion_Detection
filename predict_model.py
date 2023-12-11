import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


cfg = {
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
model = VGG('VGG19')

state_dict = torch.load('VGG19.pth', map_location=torch.device('cpu'))

model.load_state_dict(state_dict)
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(48),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image_path = 'face.jpeg' 
def predict_emotion(image_path):
    image = Image.open(image_path)
    image_tensor = image_transform(image).unsqueeze(0) 
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted_idx = torch.max(outputs, 1)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion = emotions[predicted_idx.item()]

    return predicted_emotion

print(predict_emotion(image_path))
