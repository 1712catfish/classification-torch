try:
    INTERACTIVE
except Exception:
    from setup_configs import *


class MayoDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root, df, transform=None, target_transform=None):
        self.root = root
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.data = self.df[['image_id', 'label']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id, label = self.data[index]

        image = Image.open(os.path.join(self.root, image_id + '.jpg'))
        if self.transform is not None:
            image = self.transform(image)

        label = S2I_LBL_MAP[label]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
