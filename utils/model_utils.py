from other_imports import *
from configs import *

class ModelUtils(ConfigSelector):
    def __init__(self):    
        super(ModelUtils, self).__init__()

    # LOAD MODEL #
    def load_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model
 
    def count_parameters(self, model):
        trainable, frozen  = 0, 0
        table = PrettyTable(["Modules", "Parameters", "Requires grad"])    
        
        for name, parameter in model.named_parameters():
            param = parameter.numel()
            table.add_row([name, param, parameter.requires_grad])
    
            if parameter.requires_grad:  trainable+=param
            else: frozen+=param
                
        percent_frozen = round(100*frozen/(trainable+frozen),4)
        percent_trainable = round(100*trainable/(trainable+frozen),4)
    
        print(table)
        print('============================================')
        print("Total Params: {}".format(trainable+frozen))
        print("Total Trainable Params: {}".format(trainable))
        print("Total Non-Trainable Params: {}".format(frozen))
        print("Percent frozen: {}".format(percent_frozen), "%")
        print("Percent trainable: {}".format(percent_trainable), "%")
        print('=============================================')

    
    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
    
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(True)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(True)
                module.train()
    
        return model

    def freeze_only_backbone(self, model):
        for param in model.backbone.parameters():
            param.requires_grad = False

        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()        
        return model
    

    def fetch_scheduler(self, optimizer, *args):
        
        if self.sched == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=self.T_0, 
                                                                 T_mult=2, 
                                                                 eta_min=self.min_lr,
                                                                 last_epoch=-1)
        elif self.sched == "OneCycle":
            dl = args[0]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                            max_lr=1e-4,
                                                            steps_per_epoch=len(dl),
                                                            epochs=self.epochs,
                                                            pct_start=0.25)
        
        elif self.sched == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=self.step_size, 
                                                        gamma=self.gamma) 


        elif self.sched is None:
            return None
        
        else:
            raise NotImplementedError(f"{self.sched} is not yet implemented!")
        
        return scheduler

    def select_criterion(self):
        if self.crit == "BCE":
            criterion = nn.BCEWithLogitsLoss() 
        else:
            raise NotImplementedError(f"{self.crit} is not yet implemented!")
        return criterion

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)
