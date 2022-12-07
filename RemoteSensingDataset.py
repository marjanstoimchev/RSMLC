from other_imports import *
from configs import *

   
class BEN_Dataset(torch.utils.data.Dataset):

    def __init__ (self, df, transforms=None):
        
        super(BEN_Dataset, self).__init__()
        self.df = df
        self.transforms = transforms
        self.data = [(row[0], np.array(list(row[1:]))) for _, row in self.df.iterrows()]

    def __len__(self):
        return len(self.df)

    def load_patches(self, patch_dir):
        _OPTICAL_MAX_VALUE = 2000.
        bands = [np.asarray(
        Image.open("../BigEarthNet-v1.0" + "/" + f"{patch_dir}" + "/" + f"{patch_dir}_B{band}.tif"), dtype=np.uint16) for band in ["04", "03", "02"]]
        stacked_arr = np.stack(bands, axis=-1)
        image = stacked_arr  /_OPTICAL_MAX_VALUE * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def __getitem__(self, index):
        
        image_path, label = self.data[index]             
        image = self.load_patches(image_path)

        if self.transforms:
            image = self.transforms(image=image)["image"] / 255.0

        return {"image": image, 
                "label": torch.as_tensor(label)}  

class RSDataset(torch.utils.data.Dataset):

    def __init__ (self, df, image_size, transforms=None):
        super(RSDataset, self).__init__()
        
        self.df = df
        self.image_size = image_size
        self.transforms = transforms
        self.data = [(row[0], row[1:]) for _, row in self.df.iterrows()]
    
    def __len__(self):
        return len(self.df)

    def __getitem__ (self, index):
        
        image_path, label = self.data[index]  
        label = np.array(list(label.values))
        label = torch.from_numpy(label)
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation = cv2.INTER_CUBIC)

        if self.transforms:
            image = self.transforms(image=image)["image"] / 255.0            
        
        return {"image": image, 
                "label": torch.as_tensor(label)}    
   
