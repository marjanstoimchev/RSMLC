from utils import *
from other_imports import * 
from configs import ConfigSelector

class FeatureExtractionTask(ModelUtils):
    
    """
    Class for the feature extraction task. 
    
    :param model         d_type (obj) : pytorch model
    :param model_name    d_type (str) : type of network architecture either -> Vgg16, Vgg19, resnet34, resnet50, resnet152, effnet_b0, effnet_b1, effnet_b2
    :param dataloaders   d_type (dict): dictionary of dataloaders, with keys: {train, test, val}
    :param dataset       d_type (str) : dataset name
    :param path_features d_type (str) : path where to store the features
    :param path_model    d_type (str) : path where to store the model  
    :param device        d_type (obj) : used device, either cpu, or cuda
    :param config        d_type (obj) : subclass to select the dataset config based on dataset name
    """
    
    
    def __init__(self, model, model_name, dataset, dataloaders, device, 
                 path_features,
                 feature_type = "FineTune",
                 path_model   = "saved_models"):
        
        super(FeatureExtractionTask, self).__init__()
        self.model_name    = model_name
        self.dataloaders   = dataloaders
        self.dataset       = dataset
        self.path_features = path_features
        self.feature_type = feature_type
        self.path_model    = path_model
        
        self.device        = device
        self.config        = self.select(self.dataset) # the use of select method from the ConfigSelector to access the configs for the respective dataset

        PATH = create_path(f"{path_model}/{self.feature_type}/{self.dataset}/{self.model_name}")
        models_paths = SortedAlphanumeric(os.listdir(PATH)).sort()
        self.path_model = PATH + '/' + models_paths[-1]
                
        info_message("LOADING MODEL: {}", self.path_model)
        self.model = self.load_model(model,   self.path_model) 
        
    @torch.inference_mode()
    def run(self):
        
        """
        Method that will run the feature extraction task 
        :return: 
        """
        self.model.eval()
      
        info_message("\n")
        info_message("Dataframes: {}", self.path_features)
        info_message("\n")
        
        # iterating through train, test and val dataloaders 
        for data_type, dl in self.dataloaders.items():
            pbar = tqdm(enumerate(dl), total=len(dl),
                        desc=f"Extracting features for ---{data_type} set ---: ", position=0, leave=True)
    
            features = []
            targets = []
          
            for step, data in pbar:
                x = data['image'].to(self.device, dtype=torch.float)
                y = data['label'].to(self.device, dtype=torch.float) 
                features += [self.model.extract(x).cpu().detach().numpy()]
                targets  += [y.cpu().detach().numpy()]
              
            # flattening along number of instances dimension
            targets  = np.array(list(itertools.chain.from_iterable(targets))).astype(int) # dim: (num_features, num_examples)
            features = np.array(list(itertools.chain.from_iterable(features)))            # dim: (num_targets,  num_examples)
    
            info_message("{} Features shape: {}", data_type, features.shape)
            info_message("{} Targets shape: {}",  data_type, targets.shape)
                            
            # full paths to store the features and targets
            full_path_features = "/".join([self.path_features, f"{data_type}_features.npy"])
            full_path_targets = "/".join([self.path_features, f"{data_type}_targets.npy"])
                                          
            
            # save to npy file
            save(full_path_features, features)
            save(full_path_targets,  targets)
            
            
    
