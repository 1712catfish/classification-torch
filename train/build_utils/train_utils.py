try:
    INTERACTIVE
except Exception:
    from setup_configs import *


def get_pretrain_efficient(name, checkpoint=None):
    model = EfficientNet.from_name(name)
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
    return model


def run_one_epoch(model, loader, steps, optimizer, criterion, train=True):
    assert not (train and criterion is None)

    model.train() if train else model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for image_batch, label_batch in iter(loader):
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = model(image_batch)
            loss = criterion(output, label_batch)

            if train:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * len(output)
            _, preds = torch.max(output, 1)
            epoch_acc += torch.sum(preds == label_batch.data)

    data_size = len(loader.dataset)
    epoch_loss = epoch_loss / data_size
    epoch_acc = epoch_acc / data_size

    if train:
        return model, optimizer, epoch_loss, epoch_acc
    else:
        return epoch_loss, epoch_acc


def train(model, train_loader, val_loader,
          epochs, steps_per_epoch, validation_steps,
          criterion, optimizer):
    best_acc = 0.0
    history = dict.fromkeys(['loss', 'acc', 'val_loss', 'val_acc'], [])
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model.cuda()

        model, optimizer, train_loss, train_acc = run_one_epoch(model=model,
                                                                loader=train_loader,
                                                                steps=steps_per_epoch,
                                                                optimizer=optimizer,
                                                                criterion=criterion,
                                                                train=True)
        print(f'Training loss: {train_loss:.4f | Training accuracy: {train_acc:.4f}}')

        val_loss, val_acc = run_one_epoch(model=model,
                                          loader=val_loader,
                                          steps=validation_steps,
                                          optimizer=None,
                                          criterion=criterion,
                                          train=False)
        print(f'Evaluation loss: {val_loss:.4f | Training accuracy: {val_acc:.4f}}')

        if val_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 512, 512))
            traced.save('efficientnet_model.pth')
            best_acc = val_acc

        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
