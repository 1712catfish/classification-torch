try:
    INTERACTIVE
except Exception:
    from build_utils.train_utils import *

model = get_pretrain_efficient(BACKBONE, checkpoint=CHECKPOINT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE)


