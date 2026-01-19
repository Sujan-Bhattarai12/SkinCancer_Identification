import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm.auto import tqdm

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_paths = []
        
        print(" Caching image paths to optimize speed...")
        for img_id in tqdm(self.df['image_id'], desc="Mapping Files"):
            p1 = os.path.join(data_dir, 'HAM10000_images_part_1', f'{img_id}.jpg')
            p2 = os.path.join(data_dir, 'HAM10000_images_part_2', f'{img_id}.jpg')
            # Check existence once here, not during training
            self.image_paths.append(p1 if os.path.exists(p1) else p2)
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Open and convert to RGB
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.df.loc[idx, 'label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label