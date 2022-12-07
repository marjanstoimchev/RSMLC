from other_imports import * 
from configs import *
from models import *
from base import BaseInference, BaseTrainer, BaseExtractor
from utils import SortedAlphanumeric, get_default_device, DatasetUtils, seed_all, display_label_distribution

def main(config):
    
    # setting the seed number #
    seed_all(config.seed)
    
    # Getting the device #
    device         = get_default_device()
    
    # Definition of the base tasks #
    base_trainer   = BaseTrainer(config.dataset, device, config.feature_type)
    base_inference = BaseInference(config.dataset, device, config.feature_type)
    base_extractor = BaseExtractor(config.dataset, device, config.feature_type)
    
    # Dataset Utils #
    dsu         = DatasetUtils(config.dataset)
    dataframes  = dsu.prepare_dataframes()
    display_label_distribution(dataframes)
    dataloaders = dsu.prepare_dataloaders(dataframes, train = config.mode)
    
    # # Various deep learning network architectures #
    models_pallete = {  
                        "Vgg16":     Vgg(config.dataset,    "vgg16",           config.mode),
                        "Vgg19":     Vgg(config.dataset,    "vgg19",           config.mode),
                        "resnet34":  ResNet(config.dataset, "resnet34",        config.mode),
                        "resnet50":  ResNet(config.dataset, "resnet50",        config.mode),
                        "resnet152": ResNet(config.dataset, "resnet152",       config.mode),
                        "effnet_b0": EffNet(config.dataset, "efficientnet_b0", config.mode),
                        "effnet_b1": EffNet(config.dataset, "efficientnet_b1", config.mode),
                        "effnet_b2": EffNet(config.dataset, "efficientnet_b2", config.mode),
                     }
    
    # # run training #
    # base_trainer.run_training(config.epochs, models_pallete, dataloaders)

    
    # # run inference #
    # base_inference.run_prediction(models_pallete, dataloaders)
    
    # run feature extraction #
    base_extractor.run_extraction(models_pallete, dataloaders)
    
if __name__ == "__main__":
     main(config)    
     
     




