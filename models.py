from other_imports import *
from configs import *

config = ConfigSelector()

class ResNet(nn.Module):
    def __init__(self, dataset, model_name, pretrained=True):
        super(ResNet, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.8.2', model_name, pretrained=pretrained) 
        cf = config.select(dataset)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, cf.n_classes)

    def forward(self, images):
        features = self.extract(images)
        logits = self.classifier(features)
        return logits
    
    def extract(self, images):
        features = self.backbone(images)
        return features

class EffNet(nn.Module):
    def __init__(self, dataset, model_name, pretrained=True):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model(f"tf_{model_name}_ns", pretrained=pretrained)
        cf = config.select(dataset)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features, cf.n_classes)

    def forward(self, images):
        features = self.extract(images)
        logits = self.classifier(features)
        return logits
    
    def extract(self, images):
        features = self.backbone(images)
        return features

class Vgg(nn.Module):
    def __init__(self, dataset, model_name, pretrained=True):
        super(Vgg, self).__init__()
        cf = config.select(dataset)
        self.backbone =  torch.hub.load('pytorch/vision:v0.8.2', model_name, pretrained=pretrained)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.classifier = nn.Linear(in_features, cf.n_classes)
        
    def forward(self, images):
        features = self.extract(images)
        logits = self.classifier(features)
        return logits
    
    def extract(self, images):
        features = self.backbone(images)
        return features
    
