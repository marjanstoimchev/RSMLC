import argparse

# BigEarthNet labels and mappings #

original = {
        "Continuous urban fabric": 0,
        "Discontinuous urban fabric": 1,
        "Industrial or commercial units": 2,
        "Road and rail networks and associated land": 3,
        "Port areas": 4,
        "Airports": 5,
        "Mineral extraction sites": 6,
        "Dump sites": 7,
        "Construction sites": 8,
        "Green urban areas": 9,
        "Sport and leisure facilities": 10,
        "Non-irrigated arable land": 11,
        "Permanently irrigated land": 12,
        "Rice fields": 13,
        "Vineyards": 14,
        "Fruit trees and berry plantations": 15,
        "Olive groves": 16,
        "Pastures": 17,
        "Annual crops associated with permanent crops": 18,
        "Complex cultivation patterns": 19,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 20,
        "Agro-forestry areas": 21,
        "Broad-leaved forest": 22,
        "Coniferous forest": 23,
        "Mixed forest": 24,
        "Natural grassland": 25,
        "Moors and heathland": 26,
        "Sclerophyllous vegetation": 27,
        "Transitional woodland/shrub": 28,
        "Beaches, dunes, sands": 29,
        "Bare rock": 30,
        "Sparsely vegetated areas": 31,
        "Burnt areas": 32,
        "Inland marshes": 33,
        "Peatbogs": 34,
        "Salt marshes": 35,
        "Salines": 36,
        "Intertidal flats": 37,
        "Water courses": 38,
        "Water bodies": 39,
        "Coastal lagoons": 40,
        "Estuaries": 41,
        "Sea and ocean": 42
    }

non_existing = [3, 4, 5, 6, 7 ,8, 9, 10, 30, 32, 37]

labels_19 = {
        "Urban fabric": 0,
        "Industrial or commercial units": 1,
        "Arable land": 2,
        "Permanent crops": 3,
        "Pastures": 4,
        "Complex cultivation patterns": 5,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
        "Agro-forestry areas": 7,
        "Broad-leaved forest": 8,
        "Coniferous forest": 9,
        "Mixed forest": 10,
        "Natural grassland and sparsely vegetated areas": 11,
        "Moors, heathland and sclerophyllous vegetation": 12,
        "Transitional woodland, shrub": 13,
        "Beaches, dunes, sands": 14,
        "Inland wetlands": 15,
        "Coastal wetlands": 16,
        "Inland waters": 17,
        "Marine waters": 18
    }

new_mappings =  {0: [0, 1],
                 1: [2],
                 2: [11,12,13],
                 3: [14, 15, 16, 18],
                 4: [17],
                 5: [19],
                 6: [20],
                 7: [21],
                 8: [22],
                 9: [23],
                 10: [24],
                 11: [25, 31],
                 12: [26, 27],
                 13: [28],
                 14: [29],
                 15: [33, 34],
                 16: [35, 36],
                 17: [38, 39],
                 18: [40, 41, 42]
             
             }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class BaseConfig(object):
    num_workers = 4
    fold_num = 5
    seed = 42
    #base_dir = '../rs_datasets/mlc'
    base_dir = '../DeepLearningRS/rs_datasets'
    learning_rate = 1e-4 #1e-3
    gamma = 0.1
    weight_decay = 1e-6
    optim = 'Adam'
    crit = "BCE"
    sched = 'CosineAnnealingWarmRestarts' #step
    step_size = 10 # 25
    n_accumulate = 1 #5
    verbose_step = 1
    early_stopping = 10
    min_lr = 1e-5
    T_max = 500
    T_0 = 3
    
class UCMConfig(BaseConfig):
    extension = 'tif'
    n_classes = 17
    image_size = 256

class BENConfig19(BaseConfig):
    extension = 'tif'
    n_classes = 19
    image_size = 120
    path = "BEN_Dataset/images/"
    
class BENConfig43(BaseConfig):
    extension = 'tif'
    n_classes = 43
    image_size = 120
    path = "BEN_Dataset/images/"
    
class AIDConfig(BaseConfig):
    extension = 'jpg'
    n_classes = 17
    image_size = 600
    
class DFC15Config(BaseConfig):
    extension = 'png'
    n_classes = 8
    image_size = 600

class AnkaraConfig(BaseConfig):
    extension = 'bmp'
    extension_ms = '.mat'
    n_classes = 29
    image_size = 63

class MLRSNetConfig(BaseConfig):
    extension = 'jpg'
    n_classes = 60
    image_size = 256


class ConfigSelector(BaseConfig):
    def __init__(self):
        super(ConfigSelector, self).__init__()
        self.args = self.ConfigParser()
        self.dataset = self.args.dataset
        self.mode = self.args.mode
        self.epochs = self.args.n_epochs
        self.batch_size = self.args.batch_size
        self.feature_type = self.args.feature_type
        self.lr = self.args.lr

    def select(self, dataset):
        if dataset == 'UCM':
            config = UCMConfig()
        elif dataset == 'AID':
            config = AIDConfig()
        elif dataset == 'DFC_15':
            config = DFC15Config()
        elif dataset == 'Ankara':
            config = AnkaraConfig()
        elif dataset == 'MLRSNet':
            config = MLRSNetConfig()
        elif dataset == 'BEN_19_full':
            config = BENConfig19()
        elif dataset == 'BEN_43_full':
            config = BENConfig43()           

        return config
    
    def ConfigParser(self):

        parser = argparse.ArgumentParser(description='Process hyper-parameters')
        parser.add_argument('--dataset', type=str, default="UCM", help='Dataset')
        parser.add_argument('--mode', type=int, default=True, help='Mode')
        parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('--seed', type=int, default=42, help='Seed')
        parser.add_argument('--lr', type=int, default=1e-4, help='Learning Rate')
        parser.add_argument('--feature_type', type=str, default="Pre-Train", help='Types of features')
        args = parser.parse_args()        
        return args
            
