from other_imports import *

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
   
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path) 
    return path

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
 

class SortedAlphanumeric(object):
    def __init__(self, data):
        super(SortedAlphanumeric, self).__init__()
        self.data = data
        
    def sort(self):  
       """https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir"""
        
       convert = lambda text: int(text) if text.isdigit() else text.lower()
       alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
       return sorted(self.data, key=alphanum_key)   
 
    
def detect_presence_of_labels(dataframes):
    for df in dataframes:
        values = df.values[:, 1:]
        where = values.sum(0) == 0
        print(where.sum())

def display_label_distribution(dataframes):
    dfs = []
    for df in dataframes:
        dfs += [df.iloc[:, 1:].sum()]
    df = pd.concat(dfs, axis = 1)
    df = df.rename(columns = {i:name for i, name in enumerate(["train", "val", "test"])})
    info_message("")
    info_message(" - - - - -  Label Distribution  - - - - - ")
    
    info_message("{}", df)
    info_message(" - - - - - - - - - - - - - - - - - - - - -")

   
def info_message(message, *args, end="\n"):
    print(message.format(*args), end=end) 


def seed_all(seed: int = 1930):
    print("Using Seed Number {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)  # for numpy pseudo-random generator
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # False
    #torch.backends.cudnn.enabled = False # False
    #torch.set_num_threads(1)
    #torch.cuda.set_device(1)


def findmax(outputs):
    Max = -float("inf")
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index


def OneError(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    num = 0
    one_error = 0
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            Max, index = findmax(outputs[i])
            num = num + 1
            if test_target[i][index] != 1:
                one_error = one_error + 1
    return one_error / num


def compute_metrics(y_true, y_pred, scores):

    # Example based
    sa = accuracy_score(y_true, y_pred, normalize=True)
    hl = hamming_loss(y_true, y_pred)
    
    # Label: threshold dependent
    micro_precision = average_precision_score(y_true, scores, average="micro")
    micro_recall    = recall_score(y_true, y_pred, average='micro')
    micro_f1        = f1_score(y_true, y_pred, average='micro')
    micro_auc       = roc_auc_score(y_true, scores, average='micro')
    
    macro_precision = average_precision_score(y_true, scores, average="macro")
    macro_recall    = recall_score(y_true, y_pred, average='macro')
    macro_f1        = f1_score(y_true, y_pred, average='macro')
    
    # Ranking based
    cov             = coverage_error(y_true, y_pred)
    rl              = label_ranking_loss(y_true, scores)
    avgp            = label_ranking_average_precision_score(y_true, scores)
    oe              = OneError(y_pred, y_true)
        
    metrics = [sa, hl, 
                micro_precision, micro_recall, micro_f1, micro_auc,
                macro_precision, macro_recall, macro_f1,
                cov, rl, avgp, oe]

    return metrics


# class EvaluateRF(object):
#     def __init__(self, x_val, y_val, x_test, y_test):
                
#         self.x_val = x_val
#         self.y_val = y_val
#         self.x_test = x_test
#         self.y_test = y_test
        
    
#     def reduce_proba(self, probabilities):
#         proba = []
#         for p in probabilities:
#             if p.shape[1] == 1: 
#                 proba.append(p)
#             else:
#                 proba.append(np.expand_dims(p[:, 1], axis=1))
#         proba = np.array(proba).T
#         proba = np.squeeze(proba, 0)
#         return proba
    
#     def generate_predictions(self, tree):
    
#         proba_val = tree.predict_proba(self.x_val)
#         proba_test = tree.predict_proba(self.x_test)
        
#         proba_val = self.reduce_proba(proba_val)
#         proba_test = self.reduce_proba(proba_test)
                
#         best_threshes = [self.best_thresh(proba_val[:,i], self.y_val[:,i]) for i in tqdm(range(proba_val.shape[1]))]    
#         y_pred_test = np.array([[1 if proba_test[i,j]>=best_threshes[j] else 0 for j in range(len(best_threshes))] for i in range(len(self.y_test))]) 
#         return y_pred_test, proba_test 

# class EvaluateModel(object):
#     def __init__(self, model_inference, path, dataloader):
                
#         self.path = path
#         self.dataloader = dataloader
#         self.model_inference = model_inference
        
    
#     def get_probabilities(self, mode):
        
#         labels = [y.detach() for x,y in tqdm(self.dataloader[mode])]
#         proba = [torch.sigmoid(self.model_inference(x)).detach() for x, y in tqdm(self.dataloader[mode])]      
#         proba = torch.cat(proba, dim=0)
#         labels = torch.cat(labels, dim=0)
        
#         return proba, labels
    
#     def single_macro_fbeta(self, y_pred:Tensor, y_true:Tensor, thresh:float=0.2, 
#              beta:float=1, eps:float=1e-9):
        
#         "Computes the macro averaged f_beta between preds and targets for binary tensors"
    
#         beta2 = beta**2     
#         pred_i = (y_pred>thresh).float()
#         true_i = y_true.float()
#         tp = (pred_i*true_i).sum()
#         prec = tp/(pred_i.sum()+eps)
#         rec = tp/(true_i.sum()+eps)
#         res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2) 
        
#         return(res.mean())
    
#     def best_thresh(self, y_pred:Tensor, y_true:Tensor, precision = 10000):

#         "Computes the best threshold for macro averaged f_beta between preds and targets  for binary tensors"
        
#         thresh = []
#         result = []
#         for i in range(precision):
#             thresh.append(i/precision)
#             result.append(self.single_macro_fbeta(y_pred,y_true,thresh=i/precision))
#         idx = np.argmax(result)
        
#         return(thresh[idx])
    
#     def predict(self):
        
#         proba_val, labels_val = self.get_probabilities(mode = 'val')
#         print(proba_val.shape)
#         print(labels_val.shape)
        
#         proba_val = proba_val.cpu()
#         labels_val = labels_val.cpu()
        
#         best_threshes = [self.best_thresh(proba_val[:,i],labels_val[:,i]) for i in tqdm(range(proba_val.shape[1]))]  #range(self.n_classes)                  
#         proba_test, labels_test = self.get_probabilities(mode = 'test')
#         labels_pred_test = np.array([[1 if proba_test[i,j]>=best_threshes[j] else 0 for j in range(len(best_threshes))] for i in range(len(labels_test))])                  
#         labels_test = labels_test.detach().cpu().numpy()
#         proba_test = proba_test.detach().cpu().numpy()
        
#         return labels_test, labels_pred_test, proba_test, best_threshes
   


    



    
    
    

