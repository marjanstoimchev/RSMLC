from other_imports import *
from configs import AnkaraConfig

class AnkaraDataset(AnkaraConfig):
    def __init__(self, multispectral = False):
        super(AnkaraDataset, self).__init__()
        
        self.multispectral = multispectral
        self.images_path = self.base_dir + '/Ankara/AnkaraHSIArchive'
        self.labels_path = self.images_path + '/' + 'multilabel.txt'
        
    def load_dataframe(self):
        
        df = pd.read_csv(self.labels_path, delimiter = "\t")       
        directories = [x[0] for x in os.walk(self.images_path)][0]
                
        files = glob.glob(os.path.join(directories, '*' + self.extension)) 
        df['IMAGE\LABEL'] = files
        
        if self.multispectral:
            df_ms = df.copy()
            files_multispectral = glob.glob(os.path.join(directories, '*' + self.extension_ms))
            df_ms['IMAGE\LABEL'] = files_multispectral
            df = df_ms
            df = df.loc[df.values[:, 1:].sum(1) != 0]
            
        return df
   
