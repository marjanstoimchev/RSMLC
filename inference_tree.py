import argparse
from utils import *
from other_imports import * 

def main(config):
    
    model_names = ['Vgg16']
                   #, 'Vgg19', 'resnet34', 'resnet50', 'resnet152', 'effnet_b0', 'effnet_b1', 'effnet_b2']

    metric_names = ["subset accuracy", "hamming loss",                           # example based
                    "micro precision", "micro recall", "micro f1", "micro auc",  # label based (micro)
                    "macro precision", "macro recall", "macro f1",               # label based (macro)
                    "coverage", "ranking loss", "average precision", "one-error" # ranking based
                    ]
    
    root_dir_features = "RESULTS/FEATURES"
    root_dir_metrics  = "RESULTS/METRICS"
    
    ensemble = Tree(n_estimators   = config.n_estimators,
                    max_features   = config.max_features,
                    tr_type        = config.tr_type,
                    seed           = config.seed,
                    pred_threshold = 0.5)
    
    ensemble.run_multi_prediction(root_dir_features, 
                                  root_dir_metrics, 
                                  config.feature_type,
                                  model_names, 
                                  metric_names, 
                                  config.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--dataset',      type=str, default="UCM", help='Dataset')
    parser.add_argument('--feature_type', type=str, default="Pre-Train", help='Type of features')
    parser.add_argument('--tr_type',      type=str, default="forest", help='Type of tree ensemble')
    parser.add_argument('--n_estimators', type=int, default=150, help='Number of estimators')
    parser.add_argument('--max_features', type=str, default="sqrt", help='Feature subsed size')
    parser.add_argument('--seed', type=int, default=42, help='Seed number')
    config = parser.parse_args()
    main(config)
