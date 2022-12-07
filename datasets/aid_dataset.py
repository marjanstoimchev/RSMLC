from other_imports import *
from configs import AIDConfig
from utils.other_utils import SortedAlphanumeric


class AIDDataset(AIDConfig):
    def __init__(self):
        
        super(AIDDataset, self).__init__()
        self.path_train = self.base_dir + '/AID_Dataset/images/images_tr'
        self.path_test = self.base_dir + '/AID_Dataset/images/images_test'
        self.labels_path = self.base_dir + '/AID_Dataset/multilabel.csv'

    
    def load_dataframe(self):
        
        df = pd.read_csv(self.labels_path)
        label_names = list(df.keys()[1:])
        
        dir_train = [x[0] for x in os.walk(self.path_train)][1:]
        dir_test = [x[0] for x in os.walk(self.path_test)][1:]
            
        files_train = [d.split('.' + self.extension)[0] for dirs in dir_train for d in os.listdir(dirs)]
        files_test = [d.split('.' + self.extension)[0] for dirs in dir_test for d in os.listdir(dirs)]

        files_train = SortedAlphanumeric(files_train).sort()
        files_test = SortedAlphanumeric(files_test).sort()
        
        targets_train = [int(re.split(r'(\d+)', file.split('/')[-1])[1]) - 1 for file in files_train]
        targets_test = [int(re.split(r'(\d+)', file.split('/')[-1])[1]) - 1 for file in files_test]

        df_train = df[df['IMAGE/LABEL'].isin(files_train)]
        df_test = df[df['IMAGE/LABEL'].isin(files_test)]
        
        f_tr = [dirs + '/' for dirs in dir_train for d in os.listdir(dirs)]
        f_te = [dirs + '/' for dirs in dir_test for d in os.listdir(dirs)]
        
        f_tr = SortedAlphanumeric(f_tr).sort()
        f_te = SortedAlphanumeric(f_te).sort()
        
        Df_train = df_train.copy()
        Df_test = df_test.copy()
        
        Df_train['IMAGE/LABEL']  = f_tr + Df_train['IMAGE/LABEL'] + '.' + self.extension
        Df_test['IMAGE/LABEL']  =  f_te + Df_test['IMAGE/LABEL']  + '.' + self.extension
        return Df_train, Df_test
