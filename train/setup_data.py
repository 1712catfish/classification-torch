import pandas as pd

try:
    INTERACTIVE
except Exception:
    from build_utils.data_utils import *

df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label)

train_loader = torch.utils.data.DataLoader(
    MayoDataset(root=TRAIN_DIR, df=train_df,
                transform=None,
                target_transform=None),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    MayoDataset(root=TRAIN_DIR, df=val_df,
                transform=None,
                target_transform=None),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

STEPS_PER_EPOCH = len(train_df) // BATCH_SIZE
VALIDATION_STEPS = len(val_df) // BATCH_SIZE + 1
