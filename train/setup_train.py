try:
    INTERACTIVE
except Exception:
    from build_utils.train_utils import *

model = build_efficient_net(BACKBONE, num_classes=N_CLASSES, checkpoint=CHECKPOINT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE)
