from other_imports import *
from utils import *
from configs import *
from tasks import Trainer

class BaseTrainer(ConfigSelector):
    def __init__(self, dataset, device, feature_type):                                  
        super(BaseTrainer, self).__init__() 
        self.dataset = dataset
        self.device = device
        self.feature_type = feature_type
    
    def run_training(self, epochs, models_pallete, dataloaders, model_path = "saved_models", stats_path = "RESULTS/STATS"):
        
        # - - - - - - Model definition - - - - - - #
        ranking, example, label = [], [], []
        model_names_dl = []
        STATS = {}
        for model_name, model in models_pallete.items():
            # - - - - - - - - - - - - - - - - - - - - - - - - #
            info_message("")
            info_message("---{} MODEL TRAINING---", model_name)
            info_message("model path: {}", model_path)
            info_message("stats path: {}", stats_path)
            info_message("Batch size: {}", self.batch_size)
            info_message("Types of features: {}", self.feature_type)
            info_message("Initial lr: {}", self.lr)
            info_message("")
            
            # Model Utils #
            model_utils = ModelUtils()
            model = to_device(model, self.device)
            
            if   self.feature_type == "FineTune": model = model
            elif self.feature_type == "Pre-Train": model = model_utils.freeze_only_backbone(model)
                
            params = filter(lambda p: p.requires_grad, model.parameters())
            criterion = model_utils.select_criterion()

            optimizer = optim.Adam(params, lr=self.learning_rate)
            scheduler = model_utils.fetch_scheduler(optimizer)
            
            path_models = create_path(f"{model_path}/{self.feature_type}/{self.dataset}/{model_name}")
            path_stats  = create_path(f"{stats_path}/{self.feature_type}/{self.dataset}")

            trainer = Trainer(model, self.device, criterion, optimizer, scheduler)
            history = trainer.fit(epochs, dataloaders, path_models)
            
            df_stats = pd.DataFrame(history)
            df_stats.to_csv(f"{path_stats}/{model_name}.csv")

            del optimizer, model, scheduler
            torch.cuda.empty_cache() 
            gc.collect()