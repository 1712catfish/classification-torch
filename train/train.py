try:
    INTERACTIVE
except Exception:
    from setup_data import *
    from setup_train import *

train(model=model,
      train_loader=train_loader,
      val_loader=val_loader,
      epochs=12,
      steps_per_epoch=STEPS_PER_EPOCH,
      validation_steps=VALIDATION_STEPS,
      criterion=criterion,
      optimizer=optimizer)
