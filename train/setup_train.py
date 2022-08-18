try:
    INTERACTIVE
except Exception:
    from build_utils.train_utils import *

model = get_pretrain_efficient(
    'efficientnet-b4',
    checkpoint='/kaggle/input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth'
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


