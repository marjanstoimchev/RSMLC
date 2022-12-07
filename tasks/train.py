from other_imports import *
from configs import *
from utils import *

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer(ConfigSelector):
    
    
    """
    Class for training and validating the model. 
    
    :param model         d_type (obj)                               : pytorch model
    :param device        d_type (obj)                               : used device, either cpu, or cuda
    :param criterion     d_type (torch.nn.modules.loss module)      : the Binary Cross Entropy With Logits Loss for the Multi-Label Classification (MLC) task.
    :param optimizer     d_type (torch.optim.adam module)           : the Adam otpimizer
    :param scheduler     d_type (torch.optim.lr_scheduler module)   : learning rate scheduler
    :param scaler        d_type (torch.cuda.amp.grad_scaler module) : gradient scaler
    
    """
    
    
    def __init__(self, model, device, criterion, optimizer, scheduler):
        super(Trainer, self).__init__()
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.lastmodel = None
        self.best_loss = 1e3

    def save_model(self, n_epoch, save_path, loss):
        self.lastmodel = f"{save_path}/epoch_{n_epoch}-loss_{loss:.4f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            self.lastmodel,
        )

    def fit(self, epochs, dataloaders, model_path):
        
                
        """
        Method that will fit the model and will perform the training and validation
        :epochs          (int): number of epochs
        :dataloaders     (dict): dictionary of dataloaders
        :model_path      (str): path where to store the learned model
        
        :return: history (dict): training and validation history
        """
        
        gc.collect()
        not_improved_cnt = 0
        history = defaultdict(list)
        
        if torch.cuda.is_available():
            self.info_message("[INFO] Using GPU: {}", torch.cuda.get_device_name())
            
        start = time.time()
        for epoch in range(epochs):            
            train_loss, train_f1_micro, train_time = self.train_one_epoch(epoch,
                                                                            dataloaders['train'], 
                                                                            scheduler=self.scheduler, 
                                                                            schd_batch_update=True)
                
                
            valid_loss, valid_f1_micro, valid_time = self.valid_one_epoch(epoch,
                                                                            dataloaders['val'],
                                                                            scheduler=None, 
                                                                            schd_loss_update=False)
                      
            '''
            # uncomment for saving the best model at certain epoch
            if valid_loss < self.best_loss:
                self.save_model(epoch, model_path, valid_loss)
                self.info_message(
                    "valid_loss improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                    self.best_loss, valid_loss, self.lastmodel
                )
                self.best_loss = valid_loss
                not_improved_cnt = 0

            elif self.early_stopping == not_improved_cnt:
                self.info_message("\nValid loss didn't improve last {} epochs.", not_improved_cnt)
                break
            else:
                not_improved_cnt += 1 
            '''

            self.info_message(" ")
            
            history['Train Loss'].append(train_loss)
            history['Train F1 micro'].append(train_f1_micro)
            
            history['Valid Loss'].append(valid_loss)
            history['Valid F1 micro'].append(valid_f1_micro)       
            
        self.save_model(epoch, model_path, valid_loss)
        
        end = time.time()
        time_elapsed = end - start
        info_message('Training complete in {:.0f}h {:.0f}m {:.0f}s', 
                        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60)
            
        self.best_loss = valid_loss
        info_message("Best Loss: {:.4f}", self.best_loss)

        del self.model, self.optimizer
        torch.cuda.empty_cache() 
        return history

    
    def train_one_epoch(self, epoch, train_loader, scheduler=None, schd_batch_update=True):  

        
        """
        Method that will perform one epoch of training
        :epoch             (int): current epoch
        :train_loader      (obj): the train dataloader
        :scheduler         (obj): scheduler
        :schd_batch_update (bool): whether to apply per batch update of the loss
        
        :return: avg_loss  (float): batch average bce loss
                 f1_micro  (float): batch average micro F1 metric
                 time      (int): training time
        """
        
        self.model.train()
        t = time.time()
        losses = AverageMeter()
        F1_micro = AverageMeter()
        
        bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f"Epoch {epoch} - Train Step: ", position=0, leave=True)
            
        for step, data in bar:
            images = data['image'].to(self.device, dtype=torch.float)
            labels = data['label'].to(self.device, dtype=torch.float)
            
                        
            with autocast():
                # predict # 
                logits = self.model(images)
                loss   = self.criterion(logits, labels)
                preds  = torch.sigmoid(logits).data > 0.5  
                batch_size = images.size(0)
        
                f1_miro = f1_score(labels.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="micro")
                losses.update(loss.item(), batch_size)    
                F1_micro.update(f1_miro.item(), batch_size)
                
                if self.n_accumulate > 1:
                    loss = loss / self.n_accumulate
                    
                # Scaling the loss #
                self.scaler.scale(loss).backward()

                # Gradient accumulation #
                if ((step + 1) %  self.n_accumulate == 0) or ((step + 1) == len(train_loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
               
                    self.optimizer.zero_grad()     
                    if self.scheduler is not None and schd_batch_update:
                        self.scheduler.step()

                # calculate running loss #  
                if ((step + 1) % self.verbose_step == 0) or ((step + 1) == len(train_loader)):                    
                    bar.set_postfix(epoch_loss=losses.avg,
                                    f1_micro=F1_micro.avg,
                                    LR=self.optimizer.param_groups[0]['lr']) 

                    
        if self.scheduler is not None and not schd_batch_update:
            self.scheduler.step()
        gc.collect()
        
        return losses.avg, F1_micro.avg, int(time.time() - t)
        
    @torch.inference_mode()
    def valid_one_epoch(self, epoch, val_loader, scheduler=None, schd_loss_update=False):


        """
        Method that will perform the validation on epoch end
        :epoch             (int): current epoch
        :val_loader        (obj): the validation dataloader
        :scheduler         (obj): scheduler
        :schd_batch_update (bool): whether to apply per batch update of the loss
        
        :return: avg_loss  (float): batch average bce loss
                 f1_micro  (float): batch average micro F1 metric
                 time      (int): training time
        """

        self.model.eval()
        t = time.time()
        dataset_size = 0
        running_loss = 0.0
        losses = AverageMeter()
        F1_micro = AverageMeter()
        
        bar = tqdm(enumerate(val_loader), total=len(val_loader),
                                desc=f"Epoch {epoch} - Valid Step: ", position=0, leave=True)


        for step, data in bar:    
            images = data['image'].to(self.device, dtype=torch.float)
            labels = data['label'].to(self.device, dtype = torch.float)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            preds = torch.sigmoid(logits).data > 0.5  
            batch_size = images.size(0)
            
            f1_miro = f1_score(labels.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="micro")
            losses.update(loss.item(), batch_size) 
            
            F1_micro.update(f1_miro.item(), batch_size)
            if ((step + 1) % self.verbose_step == 0) or ((step + 1) == len(val_loader)):
                bar.set_postfix(epoch_loss=losses.avg,
                                f1_micro=F1_micro.avg,
                                LR=self.optimizer.param_groups[0]['lr']) 

                
        if self.scheduler is not None:
            if schd_loss_update:
                self.scheduler.step(losses.avg)
            else:
                self.scheduler.step()
        
        gc.collect()
        return losses.avg, F1_micro.avg, int(time.time() - t)
        
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)
