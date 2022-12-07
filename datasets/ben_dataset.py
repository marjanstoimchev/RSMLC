from other_imports import *
from configs import BENConfig43, BENConfig19
    
class BEN_Dataset_Full_43(BENConfig43):
    def __init__(self):
        super(BEN_Dataset_Full_43, self).__init__()
        self.path_df = self.base_dir + f"multi_hot_labels_{str(self.n_classes)}.txt"

    def load_dataframe(self):
        df = pd.read_csv(self.path_df, index_col = 0, sep = "\t")
        return df

class BEN_Dataset_Full_19(BENConfig19):
    def __init__(self):
        super(BEN_Dataset_Full_19, self).__init__()
        self.path_df = self.base_dir + f"multi_hot_labels_{str(self.n_classes)}.txt"

    def load_dataframe(self):
        df = pd.read_csv(self.path_df, index_col = 0, sep = "\t")
        return df
