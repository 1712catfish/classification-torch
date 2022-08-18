import pandas as pd
import torch.nn.functional

try:
    INTERACTIVE
except Exception:
    from build_utils.data_utils import *

target_transfrom = lambda x: torch.nn.functional.one_hot(x, num_classes=N_CLASSES)

df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df.label)

train_loader = torch.utils.data.DataLoader(
    MayoDataset(root=TRAIN_DIR, df=train_df,
                transform=transform['train'],
                target_transform=target_transfrom),
    batch_size=BATCH_SIZE,
    shuffle=True, drop_last=True,
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

N_TRAIN, N_VAL = len(train_df), len(val_df)
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE + 1 if N_TRAIN % BATCH_SIZE > 0 else 0
VALIDATION_STEPS = N_VAL // BATCH_SIZE + 1 if N_TRAIN % BATCH_SIZE > 0 else 0

print(f'Found {N_TRAIN} training datapoints.')
print(f'Found {N_VAL} validation datapoints.')
