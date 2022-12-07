from other_imports import *
from configs import UCMConfig
from utils.other_utils import SortedAlphanumeric

class UCMDataset(UCMConfig):
    def __init__(self):
        super(UCMDataset, self).__init__()
        self.images_path = self.base_dir + '/UCMerced_LandUse'
        self.labels_path = self.images_path + '/' + 'LandUse_Multilabeled.txt'
    
    def load_dataframe(self):
        df = pd.read_csv(self.labels_path, delimiter = "\t")
        directories = [x[0] for x in os.walk(self.images_path)][1:]
        files = [file for d in directories for file in glob.glob(os.path.join(d, '*' + self.extension))]
        files = SortedAlphanumeric(files).sort()
        df['IMAGE\LABEL'] = files
        
        return df


