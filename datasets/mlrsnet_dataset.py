from other_imports import *
from configs import MLRSNetConfig
from utils.other_utils import SortedAlphanumeric

def join_csv_labels():

    root_folder = "../rs_datasets/MLRSNet/labels"
    names = ['airplane',
        'airport',
        'bare soil',
        'baseball diamond',
        'basketball court',
        'beach',
        'bridge',
        'buildings',
        'cars',
        'chaparral',
        'cloud',
        'containers',
        'crosswalk',
        'dense residential area',
        'desert',
        'dock',
        'factory',
        'field',
        'football field',
        'forest',
        'freeway',
        'golf course',
        'grass',
        'greenhouse',
        'gully',
        'habor',
        'intersection',
        'island',
        'lake',
        'mobile home',
        'mountain',
        'overpass',
        'park',
        'parking lot',
        'parkway',
        'pavement',
        'railway',
        'railway station',
        'river',
        'road',
        'roundabout',
        'runway',
        'sand',
        'sea',
        'ships',
        'snow',
        'snowberg',
        'sparse residential area',
        'stadium',
        'swimming pool',
        'tanks',
        'tennis court',
        'terrace',
        'track',
        'trail',
        'transmission tower',
        'trees',
        'water',
        'wetland',
        'wind turbine']

    all_csv_filenames = [root_folder + "/" + r for r in  os.listdir(root_folder)]
    print(all_csv_filenames)
    
    paths, labels = [], []
    for r in  os.listdir(root_folder):
        path = root_folder + "/" + r
        df = pd.read_csv(path)
        path, label = df.values[:, :1], df.values[:, 1:]
        paths.append(path)
        labels.append(label)
        
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)
    Df = pd.DataFrame(labels)
    Df = Df.rename(columns={i:name for i, name in enumerate(names)})
    Df.insert(0, "IMAGE\LABEL", paths, True)
    
    Df.to_csv(
        "{}/multilabel.txt".format(root_folder),
        index=False,
        sep="\t",
        encoding="utf-8",
        )
    

class MLRSNetDataset(MLRSNetConfig):
    def __init__(self):
        super(MLRSNetDataset, self).__init__()
        self.images_path = self.base_dir + '/MLRSNet'
        self.labels_path = self.images_path + '/labels/' + 'multilabel.txt'

    
    def load_dataframe(self):
        df = pd.read_csv(self.labels_path, delimiter = "\t")
        directories = [x[0] for x in os.walk(self.images_path)][1:]
        files = [file for d in directories for file in glob.glob(os.path.join(d, '*' + self.extension))]
        files = SortedAlphanumeric(files).sort()
        df['IMAGE\LABEL'] = files
        df = df.loc[df.values[:, 1:].sum(1) != 0]
        return df


