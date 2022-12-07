from other_imports import *
from datasets.ankara_dataset import AnkaraDataset
from datasets.ucm_dataset import UCMDataset
from datasets.aid_dataset import AIDDataset
from datasets.dfc_15_dataset import DFC15Dataset
from datasets.mlrsnet_dataset import MLRSNetDataset
from datasets.ben_dataset import BEN_Dataset_Full_43, BEN_Dataset_Full_19


class DatasetSelector:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
  
    def select(self):
        if self.dataset_name == 'Ankara':
            ds = AnkaraDataset()
        elif self.dataset_name == 'UCM':
            ds = UCMDataset()
        elif self.dataset_name == 'AID': 
            ds = AIDDataset()
        elif self.dataset_name == 'DFC_15':
            ds = DFC15Dataset()      
        elif self.dataset_name == 'MLRSNet':
            ds = MLRSNetDataset()    
        elif self.dataset_name == "BEN_43_full":
            ds = BEN_Dataset_Full_43()  
        elif self.dataset_name == "BEN_19_full":
            ds = BEN_Dataset_Full_19()     
        return ds
 
    def generate(self):
        ds = self.select()
  
        if self.dataset_name in ['DFC_15', 'AID']:
            df, df_test = ds.load_dataframe()
            table = PrettyTable()
            table.field_names = ["Dataset", "Train", "Test"]            
            table.add_row([self.dataset_name, len(df), len(df_test)])

        else:
            df = ds.load_dataframe()
            table = PrettyTable()
            table.field_names = ["Dataset", "Train"]
            table.add_row([self.dataset_name, len(df)])
            df_test = None
        print(table)
        return ds, df, df_test


    def calculate_density_cardinality(self):

        ds = self.select()
        df_tr = ds.load_dataframe()
    
        if isinstance(df_tr, tuple):
            df_tr = pd.concat(df_tr)

        sum_Y = df_tr.iloc[:, 1:].sum().sum()
        N = df_tr.iloc[:, 1:].shape[0]
        L = max(df_tr.iloc[:, 1:].sum(1).values)
        table = PrettyTable()
        table.field_names = ["Dataset", "Cardinality", "Density"]
        table.add_row([self.dataset_name, 
                       round(sum_Y / N, 3), 
                       round((sum_Y / L) / N, 3)])
        print(table)