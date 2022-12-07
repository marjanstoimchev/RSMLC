from other_imports import * 
from utils import *
from configs import ConfigSelector

class InferenceTask(ModelUtils):
    
    """
    Class for the inference task. 
    
    :param model         d_type (obj) : pytorch model
    :param model_name    d_type (str) : type of network architecture either -> Vgg16, Vgg19, resnet34, resnet50, resnet152, effnet_b0, effnet_b1, effnet_b2
    :param dataloaders   d_type (dict): dictionary of dataloaders, with keys: {train, test, val}
    :param dataset       d_type (str) : dataset name
    :param device        d_type (obj) : used device, either cpu, or cuda
    :param feature_type  d_type (str) : type of features
    :param model_path    d_type (str) : path where to store the model  
    :param config        d_type (obj) : subclass to select the dataset config based on dataset name
    """
        
    def __init__(self, model, model_name, dataset, dataloaders, device, 
                 feature_type = "FineTune",
                 model_path   = "saved_models"
                 ):
        
        super(InferenceTask, self).__init__()

        self.model_name    = model_name
        self.dataloaders   = dataloaders
        self.dataset       = dataset
        self.device        = device
        self.feature_type  = feature_type
        self.model_path    = model_path
        self.config        = self.select(self.dataset) # the use of select method from the ConfigSelector to access the configs for the respective dataset

        PATH = create_path(f"{self.model_path}/{self.feature_type}/{self.dataset}/{self.model_name}")
        models_paths = SortedAlphanumeric(os.listdir(PATH)).sort()
        paths = PATH + '/' + models_paths[-1]
        
        info_message("LOADING MODEL: {}", paths)
        self.model = self.load_model(model, paths) 

    @torch.inference_mode()
    def predict(self):
        self.model.eval()
        
        pbar = tqdm(enumerate(self.dataloaders['test']), total=len(self.dataloaders['test']),
                    desc=f"Making predictions for ---test set ---: ", position=0, leave=True)
        
        probas, targets = [], []
        for step, data in pbar:
            logits = data['image'].to(self.device, dtype=torch.float)
            y      = data['label'].to(self.device)
            proba  = torch.sigmoid(self.model(logits)).data
           
            probas  += [proba.detach().to("cpu")]
            targets += [y.detach().to("cpu")]
        
        targets = torch.cat(targets, dim=0).to(torch.int).numpy()
        probas = torch.cat(probas, dim=0).numpy()
        preds = (probas > 0.5).astype('int')
        
        # detect all zeros in y_target #
        where_not_zero = targets.any(axis=1)
        targets = targets[where_not_zero]
        probas  = probas[where_not_zero]
        preds   = preds[where_not_zero]

        return targets, probas, preds 

