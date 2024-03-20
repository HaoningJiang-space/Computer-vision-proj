'''
此部分是VGG的实现

'''

from pyexpat import features
import torch
import torch.nn as nn


model_urls = { 
        
    'VGG11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'VGG13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
                 
              
}


cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGplus(nn.Module):
    def __init__(self, features, num_classes=1000,init_weights=False):   #1000为张平后的，后面映射为12个结果
        super(VGGplus, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), 
            nn.Linear(4096, num_classes),  #第三线性层
        ) 
        
        if init_weights:
            self._initialize_weights()
            
    def forward(self,x):
        x = self.features(x)  #N*3*224*224
        x = torch.flatten(x,start_dim=1) #N*512*7*7
        x = self.classifier(x)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
        
def make_features(cfgs:list):
    layers =[]
    in_channels = 3
    for v in cfgs:
        if v == "M":
            layers += [nn.MaxPool2d(kernal_size =2 ,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels ,v,kernel_size=3,padding = 1)
            layers += [conv2d , nn.ReLU(True)]
            in_channels = v
    
    return nn.Sequential(*layers)     
        
        
def vgg(model_name="VGG16",**kwargs):
    assert model_name in cfgs , "Warning: model number {} not in cfds dist !".format(model_name)
    cfg = cfgs[model_name]
    model = VGGplus(make_features(cfg),**kwargs)
    return model                                                                                


#test

#if __name__ == '__main__':
        
        