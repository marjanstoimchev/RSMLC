from other_imports import *
from utils import *
from configs import *
from tasks import FeatureExtractionTask

class BaseExtractor(ConfigSelector):
    def __init__(self, dataset, device, feature_type):                                  
        super(BaseExtractor, self).__init__() 
        self.dataset = dataset
        self.feature_type = feature_type     
        self.device = device

    def run_extraction(self, models_pallete, dataloaders, 
                       features_path = "RESULTS/FEATURES",
                       path_model    = "saved_models"
                       ):  
        
        
        
        for model_name, model in models_pallete.items():
                        
            path_features = "/".join([f"{features_path}", 
                                      f"{self.feature_type}", 
                                      f"{self.dataset}", 
                                      f"{model_name}"]) 
            
            path_features = create_path(path_features)
    
            info_message("\n")
            info_message("MODEL FEATURE EXTRACTION: {}", model_name)
            info_message("STORE PATH: {}", path_features)
            info_message("\n")
    
            if len(os.listdir(path_features))!=0:
                info_message("Skipping {}/{} because it already exists !!!", self.dataset, model_name)
                continue
            
            model = to_device(model, self.device)
            
            FeatureExtractionTask(model, 
                                  model_name,
                                  self.dataset,
                                  dataloaders,
                                  self.device,
                                  path_features = path_features, 
                                  feature_type  = self.feature_type,   
                                  path_model    = path_model).run()
             
            del model
            gc.collect()
            torch.cuda.empty_cache()
