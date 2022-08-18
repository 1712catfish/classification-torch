import pandas as pd
import torch.nn.functional

try:
    INTERACTIVE
except Exception:
    from build_utils.data_utils import *

df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label)

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

target_transfrom = lambda x: torch.nn.functional.one_hot(x, num_classes=N_CLASSES)

train_loader = torch.utils.data.DataLoader(
    MayoDataset(root=TRAIN_DIR, df=train_df,
                transform=transform['train'],
                target_transform=target_transfrom),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    MayoDataset(root=TRAIN_DIR, df=val_df,
                transform=transform['val'],
                target_transform=target_transfrom),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

STEPS_PER_EPOCH = len(train_df) // BATCH_SIZE
VALIDATION_STEPS = len(val_df) // BATCH_SIZE + 1

print(f'Found {len(train_df)} training datapoints.')
print(f'Found {len(val_df)} validation datapoints.')

