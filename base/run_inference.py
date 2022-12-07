from other_imports import *
from utils import *
from configs import *
from tasks import InferenceTask

class BaseInference(ConfigSelector):
    def __init__(self, dataset, device, feature_type):                                  
        super(BaseInference, self).__init__() 
        self.dataset = dataset
        self.feature_type = feature_type     
        self.device = device
    

    def run_prediction(self, models_pallete, dataloaders, 
                       model_path   = "saved_models",
                       metrics_path = "RESULTS/METRICS"
                       ):  
        
        
        metric_names = ["subset accuracy", "hamming loss",                           # example based
                        "micro precision", "micro recall", "micro f1", "micro auc",  # label based (micro)
                        "macro precision", "macro recall", "macro f1",               # label based (macro)
                        "coverage", "ranking loss", "average precision", "one-error" # ranking based
                        ]   

        
        df = pd.DataFrame(columns=list(models_pallete.keys()))
        
        for model_name, model in models_pallete.items():
            info_message("\n")
            info_message("MODEL PREDICTION: {}", model_name)
            info_message("\n")
            path_metrics = create_path(f"{metrics_path}/{self.feature_type}/{self.dataset}")
    
            if len(os.listdir(path_metrics))!=0:
                info_message("Skipping {}/{} because it already exists !!!", self.dataset, model_name)
                continue
            
            model = to_device(model, self.device)
            
            y_true, scores, y_pred  = InferenceTask(model, 
                                                    model_name,
                                                    self.dataset, 
                                                    dataloaders,
                                                    self.device,
                                                    feature_type = self.feature_type,
                                                    model_path = model_path).predict()                                               
                                                    
             
            info_message("\nshape targets: {}"    , y_true.shape) 
            info_message("shape scores: {}"       , scores.shape)
            info_message("shape predictions : {}" , y_pred.shape)
    
            metrics = compute_metrics(y_true, y_pred, scores)
            df[model_name] = metrics
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        dict = {i:name for i, name in enumerate(metric_names)}
        df = df.rename(index = dict)
        df.index.name = "metrics"
        df.to_csv(f"{path_metrics}/metrics.txt", sep = "\t")
        
        return df