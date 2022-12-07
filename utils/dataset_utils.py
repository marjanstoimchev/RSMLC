from other_imports import *
from configs import *
from datasets.dataset_selector import DatasetSelector
from RemoteSensingDataset import *
from utils.other_utils import info_message

class DatasetUtils(ConfigSelector):
    def __init__(self, dataset):
        super(DatasetUtils, self).__init__()
        self.dataset = dataset
        self.config = self.select(self.dataset)
        self.image_size = self.config.image_size
        self.ds, self.df, self.df_test = self.generate_data()
        
    def plot_batch(self, dataloader, mode):
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
        x = next(iter(dataloader[mode]))
        x = torchvision.utils.make_grid(x['image']).permute(1,2,0).detach().cpu().numpy()
        plt.imshow(x)
        plt.axis('off')
        
    def augmentations(self):
        
        data_transforms = {       
            "train": A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, border_mode=0, p=0.5),
            A.HorizontalFlip(p=0.5), 
            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3, 0.3), p=0.5),
            A.RandomSizedCrop([int(self.image_size*0.5), int(self.image_size*0.5)], # dodadeno
                                                self.image_size,
                                                self.image_size,
                                                interpolation=cv2.INTER_CUBIC, 
                                                p=0.3),
            
            A.CoarseDropout(max_holes=10, max_height=int(self.image_size * 0.05), max_width=int(self.image_size * 0.05), p=0.5),
           

            A.OneOf(
                [
                A.CLAHE(p=0.5),
                A.MotionBlur(p=0.25),
                A.MedianBlur(p=0.25),
                A.GaussianBlur(blur_limit=(3, 7), p=0.25),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
                ],
            p = 0.5,
            ), 
           
            A.CLAHE(p=0.5), 
            ToTensorV2()], p=1.), 


            
            "val": A.Compose([
            ToTensorV2()], p=1.),                 
        }    
        
        return data_transforms
    
    # Generate the train and test datasets
    def generate_data(self):
        data_select = DatasetSelector(self.dataset)
        ds, df, df_test = data_select.generate()
        data_select.calculate_density_cardinality()
        
        return ds, df, df_test
    
    # Select which dataset to use
    def select_dataset(self, df, mode):  
        
        data_transforms = self.augmentations()
        
        if self.dataset in ['BEN_43_full', 'BEN_19_full']:
            dataset = BEN_Dataset(df, transforms=data_transforms[mode])
                                  
        else:  
            dataset = RSDataset(df, self.image_size, transforms=data_transforms[mode])
               
        return dataset
    
    # Selecting specific split type for given dataframe
    def select_split_type(self, df):
        if self.dataset in ['AID', 'DFC_15']:
          splits = self.split_on_folds(df, 1, 1) # since DFC_15 and AID datasets already contain pre-determined test set that's why train_split = 1 #
        else:
            
          splits = self.split_on_folds(df, 1, 0.20) # split 25 % from train set as test set, since those datasets doesn't contain pre-deretrmined test set # # 0.2
        return splits
    
    # Select train and test dataframe 
    def select_dataframes(self, df, df_test, train_split, test_split):
        train_df = df.iloc[train_split].reset_index(drop=True)
        if df_test is not None:
            test_df = df_test
        else:
            test_df = df.iloc[test_split].reset_index(drop=True)
        return train_df, test_df    

    # Creating the fold generator
    def split_on_folds(self, df, n_splits, test_size):
        folds = df.copy()
        X, y = folds.values[:, 0], folds.values[:, 1:]
        msss = MultilabelStratifiedShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = self.seed)
        return msss.split(X, y)
    
    # Spliting the dataframes on separate sets
    def prepare_dataframes(self, fold_num = 0):
        splits = self.select_split_type(self.df) 
        for train_split, test_split in splits:
            # Initial train/test split #
            train_df, test_df = self.select_dataframes(self.df, self.df_test, train_split, test_split)
            # 0.9 % as train, 0.2 % as validation for all datasets #
            splits_train = self.split_on_folds(train_df, self.config.fold_num, 0.12) # 0.2, 15
            # Further split a portion of the train data as validation data #
            for fold, (train_split, valid_split) in enumerate(splits_train):
                new_train_df = train_df.iloc[train_split].reset_index(drop=True)
                val_df = train_df.iloc[valid_split].reset_index(drop=True)
                
                if fold == fold_num: # getting only a single train and a single validation fold #
                    break 
                
        T = sum([len(new_train_df), len(val_df), len(test_df)])
        info_message("- - - - - - - - - - - - - -  - - - - SPLITS - - - - - - - - - - - - - - - - - - - - ")
        info_message("\n{} dataset - > Train: {}/{}, Valid: {}/{} Test: {}/{}, Total: {}",
                          self.dataset,
                          len(new_train_df), round(100*len(new_train_df)/T, 3),
                          len(val_df), round(100*len(val_df)/T, 3), \
                          len(test_df), round(100*len(test_df)/T, 3),
                          T, end="\n")
        info_message("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        
        if self.dataset == "Ankara":
            test_df.loc[-1] = new_train_df.loc[new_train_df['Crop (Type-C)'] == 1].values[0]
            test_df = test_df.reset_index(drop = True)

        return new_train_df, val_df, test_df

    def prepare_dataloaders(self, dataframes, train = True):
        
        if train:
            train_ds =  self.select_dataset(dataframes[0], 'train')
            val_ds  =   self.select_dataset(dataframes[1], 'val')
            test_ds =   self.select_dataset(dataframes[2], 'val')
            status = [True, False, False]
            
        else:
            train_ds =  self.select_dataset(dataframes[0], 'val')
            val_ds  =   self.select_dataset(dataframes[1], 'val')
            test_ds =   self.select_dataset(dataframes[2], 'val') 
            status = [False, False, False]
  
        dl_train =  DataLoader(train_ds, 
                    batch_size=self.batch_size,
                    shuffle=status[0], num_workers=self.num_workers, 
                    pin_memory = True)
    
        dl_val =  DataLoader(val_ds, 
                    batch_size=self.batch_size,
                    shuffle=status[1], num_workers=self.num_workers, 
                    pin_memory = True)
                    
        dl_test =  DataLoader(test_ds, 
                    batch_size=self.batch_size,
                    shuffle=status[2], num_workers=self.num_workers, 
                    pin_memory = True)
        
    
        dataloaders = {"train": dl_train, "val": dl_val, "test": dl_test}
        return dataloaders