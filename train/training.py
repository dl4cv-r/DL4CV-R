import torch


def train_epoch(model, optimizer, loss_func, train_loader, epoch, device, verbose):
    model.train()
    torch.autograd.set_grad_enabled(True)

    # Reset to 0. There must be a better way though...
    train_loss_sum = torch.as_tensor(0.)

    for idx, (ds_imgs, labels) in enumerate(train_loader, start=1):

        ds_imgs = ds_imgs.to(device, non_blocking=True).unsqueeze(dim=1)
        labels = labels.to(device, non_blocking=True).unsqueeze(dim=1)
        residuals = labels - ds_imgs

        optimizer.zero_grad()
        pred_residuals = model(ds_imgs)
        step_loss = loss_func(pred_residuals, residuals)
        step_loss.backward()
        optimizer.step()

        with torch.no_grad():  # Necessary for metric calculation without errors due to gradient accumulation.
            train_loss_sum += step_loss

            if verbose:
                print(f'Training loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

    return train_loss_sum


def eval_epoch(model, loss_func, val_loader, epoch, device, verbose, max_imgs):
    model.eval()
    torch.autograd.set_grad_enabled(False)

    # Reset to 0. There must be a better way though...
    val_loss_sum = torch.as_tensor(0.)

    # Need to return residual and labels for residual learning.

    for idx, (ds_imgs, labels) in enumerate(val_loader, start=1):

        ds_imgs = ds_imgs.to(device, non_blocking=True).unsqueeze(dim=1)
        labels = labels.to(device, non_blocking=True).unsqueeze(dim=1)
        pred_residuals = model(ds_imgs)

        step_loss = loss_func(pred_residuals, labels - ds_imgs)
        val_loss_sum += step_loss

        if verbose:
            print(f'Validation loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

        if max_imgs:
            pass  # Implement saving images to TensorBoard.

    return val_loss_sum


def train_step(model, optimizer, loss_func, ds_imgs, labels):
    residuals = labels - ds_imgs
    optimizer.zero_grad()
    pred_residuals = model(ds_imgs)
    step_loss = loss_func(pred_residuals, residuals)
    step_loss.backward()
    optimizer.step()


def eval_step():
    pass



