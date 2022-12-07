from configs import *
from other_imports import *
from utils.other_utils import create_path, compute_metrics, info_message

def load_data(path_train, path_val, path_test):
    x_train, y_train = np.load(path_train[0]), np.load(path_train[1])
    x_val, y_tval = np.load(path_val[0]), np.load(path_val[1])
    x_test, y_test = np.load(path_test[0]), np.load(path_test[1])
    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_tval))
    return (x_train, y_train), (x_test, y_test)

class Tree(object):
    def __init__(self, n_estimators = 150,
                       max_features = "log2",
                       tr_type = "forest",
                       seed = 42,
                       pred_threshold = 0.5):
                 
        super(Tree, self).__init__()   
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.tr_type = tr_type
        self.seed = seed
        self.pred_threshold = pred_threshold


    def fit_tree(self, X):
        
        if self.tr_type == "forest":
            tree = RandomForestClassifier(n_estimators=self.n_estimators, 
                                          random_state=self.seed,
                                          max_features=self.max_features,
                                          n_jobs=-1, 
                                          verbose = 1)
        elif self.tr_type == "extra":
            
            tree = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                        random_state=self.seed,
                                        max_features=self.max_features,
                                        n_jobs=-1, 
                                        verbose = 1)
            
        return tree.fit(*X) 
        
        
    def reduce_proba(self, probabilities):
        proba = []
        for p in probabilities:
            if p.shape[1] == 1: 
                proba.append(p)
            else:
                proba.append(np.expand_dims(p[:, 1], axis=1))
        proba = np.array(proba).T
        proba = np.squeeze(proba, 0)
        return proba
    
        
    def predict(self, tree, X_test):
        scores = tree.predict_proba(X_test[0])   
        scores = self.reduce_proba(scores)
        y_pred = (scores > self.pred_threshold).astype(int)
        y_true = X_test[1].astype(int)
        return y_true, scores, y_pred

    def create_paths(self, root_dir_features, feature_type, dataset, model_name):     
        path_features = f"{feature_type}/{dataset}/{model_name}"
        info_message("PATH FEATURES: {}", path_features)
        
        features_paths = []
        targets_paths = []
        splits = ["train", "val", "test"]
        
        for split in splits:
            features_paths += ["/".join([root_dir_features, path_features, f"{split}_features.npy"])]
            targets_paths  += ["/".join([root_dir_features, path_features, f"{split}_targets.npy"])]
            
        path_train = [features_paths[0], targets_paths[0]]
        path_val   = [features_paths[1], targets_paths[1]]
        path_test  = [features_paths[2], targets_paths[2]]
        
        return path_train, path_val, path_test    
       
    def run_multi_prediction(self, root_dir_features, root_dir_metrics, feature_type, model_names, metric_names, dataset):  
        
        df = pd.DataFrame(columns=list(model_names))
        
        for model_name in model_names:
            info_message("\n")
            info_message("USING FEATURES FROM: {}", model_name)
            info_message("\n")
            path_metrics = create_path(f"{root_dir_metrics}/{feature_type}/{dataset}_{self.tr_type}_{self.max_features}")

            if len(os.listdir(path_metrics))!=0:
                self.info_message("Skipping {}/{} because it already exists !!!", dataset, model_name)
                continue
            
            path_train, path_val, path_test = self.create_paths(root_dir_features, feature_type, dataset, model_name)
            train_data, test_data = load_data(path_train, path_val, path_test)
        
            tree = self.fit_tree(train_data)
            y_true, scores, y_pred = self.predict(tree, test_data)
            metrics = compute_metrics(y_true, y_pred, scores)
            df[model_name] = metrics
            
            del tree
            gc.collect()
            
        dict = {i:name for i, name in enumerate(metric_names)}
        df = df.rename(index = dict)
        df.index.name = "metrics"
        df.to_csv(f"{path_metrics}/metrics.txt", sep = "\t")
        
        return df