from glob import glob
from torch.utils.data import Dataset, DataLoader
import os.path as osp
from PIL import Image
import torch
import cv2
from torchvision.transforms import Resize, PILToTensor
from tqdm.auto import tqdm

class ImgCapDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImgCapDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = glob(osp.join(root, '*.jpg'))

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        # img = cv2.cvtColor(cv2.imread(self.files[idx],cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = Image.open(self.files[idx])
        if self.transform:
            img = torch.from_numpy(self.transform(img).pixel_values[0])
        else:
            img = PILToTensor()(img)
            img = Resize((680, 1024), antialias=True)(img)
            if (len(img.shape) == 2):
                img = img.unsqueeze(0)
                img = img.repeat((3, 1, 1))
            if (img.shape[0] == 1):
                img = img.repeat((3, 1, 1))
            if (img.shape[0] == 4):
                img = img[:3, ...]
        assert(img.shape[0]==3), f"got {img.shape}"
        fname = osp.basename(self.files[idx])
        
        return img, fname
    

if __name__ == "__main__":

    dataset = ImgCapDataset('/workspace/img-caption/open-images/imgs/train_0')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False)
    for i, (img, fname) in enumerate(tqdm(dataloader)):
        assert (img.shape[1] == 3)
