import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter

from pathlib import Path
from time import time
import numpy as np

from utils.run_utils import initialize, get_logger, save_dict_as_json
from utils.args import create_arg_parser
from train.data_transforms import slice_normalize_and_clip
from train.checkpoints import CheckpointManager
from train.dataset import TrainingDataset
from models.unet import UnetModel


def main(args):

    ckpt_path = Path('checkpoints')
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    args.ckpt_path = ckpt_path

    log_path = Path('logs')
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    args.log_path = log_path

    save_dict_as_json(vars(save_dict_as_json), log_dir=log_path, save_name=run_name)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        args.device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Create Datasets
    train_dataset = TrainingDataset(args.train_root, transform=slice_normalize_and_clip, single_coil=False)
    val_dataset = TrainingDataset(args.val_root, transform=slice_normalize_and_clip, single_coil=False)

    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Define model, optimizer, and loss function.
    model = UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4).to(args.device, non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    loss_func = nn.L1Loss(reduction='mean').to(args.device, non_blocking=True)

    checkpointer = CheckpointManager(model, optimizer, args.save_best_only, ckpt_path, args.max_to_keep)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True, cooldown=0, min_lr=1E-7)

    writer = SummaryWriter(log_dir=str(log_path))

    example_inputs = torch.ones(size=(args.batch_size, 1, 320, 320)).to(args.device)
    writer.add_graph(model=model, input_to_model=example_inputs, verbose=False)

    previous_best = np.inf

    # Training loop. Please excuse my use of 1 based indexing here.
    logger.info('Beginning Training loop')

    for epoch in range(1, args.num_epochs + 1):
        tic = time()
        model.train()
        torch.autograd.set_grad_enabled(True)
        logger.info(f'Beginning training for Epoch {epoch:03d}')

        # Reset to 0. There must be a better way though...
        train_loss_sum = torch.as_tensor(0.)

        tot_len = len(train_loader.dataset) // args.batch_size
        for idx, (ds_imgs, labels) in tqdm(enumerate(train_loader, start=1), total=tot_len):

            ds_imgs = ds_imgs.to(args.device).unsqueeze(dim=1)
            labels = labels.to(args.device).unsqueeze(dim=1)

            optimizer.zero_grad()
            pred_residuals = model(ds_imgs)  # Predicted residuals are outputs.

            recons = pred_residuals + ds_imgs

            step_loss = loss_func(recons, labels)
            step_loss.backward()
            optimizer.step()

            with torch.no_grad():  # Necessary for metric calculation without errors due to gradient accumulation.
                train_loss_sum += step_loss

                if args.verbose:
                    print(f'Training loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

        else:
            toc = int(time() - tic)
            # Last step with small batch causes some inaccuracy but that is tolerable.
            epoch_loss = train_loss_sum.item() * args.batch_size / len(train_loader.dataset)

            logger.info(f'Epoch {epoch:03d} Training. loss: {float(epoch_loss):.4f}, Time: {toc // 60}min {toc % 60}s')

            writer.add_scalar('train_loss', epoch_loss, global_step=epoch)

        tic = time()
        model.eval()
        torch.autograd.set_grad_enabled(False)
        logger.info(f'Beginning validation for Epoch {epoch:03d}')

        # Reset to 0. There must be a better way though...
        val_loss_sum = torch.as_tensor(0.)

        # Need to return residual and labels for residual learning.
        tot_len = len(val_loader.dataset) // args.batch_size
        for idx, (ds_imgs, labels) in tqdm(enumerate(val_loader, start=1), total=tot_len):

            ds_imgs = ds_imgs.to(args.device).unsqueeze(dim=1)
            labels = labels.to(args.device).unsqueeze(dim=1)
            pred_residuals = model(ds_imgs)
            recons = pred_residuals + ds_imgs

            step_loss = loss_func(recons, labels)
            val_loss_sum += step_loss

            if args.verbose:
                print(f'Validation loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item()}')

            if args.max_imgs:
                pass  # Implement saving images to TensorBoard.

        else:
            toc = int(time() - tic)
            epoch_loss = val_loss_sum.item() * args.batch_size / len(val_loader.dataset)

            logger.info(f'Epoch {epoch:03d} Validation. loss: {float(epoch_loss):.4f} Time: {toc // 60}min {toc % 60}s')

            writer.add_scalar('val_loss', epoch_loss, global_step=epoch)

            scheduler.step(metrics=epoch_loss, epoch=epoch)  # Changes optimizer lr.

        # Checkpoint generation. Only implemented for single GPU models, not multi-gpu models.
        # All comparisons are done with python numbers, not tensors.
        if epoch_loss < previous_best:  # Assumes smaller metric is better.
            logger.info(
                f'Loss in Epoch {epoch} has improved from {float(previous_best):.4f} to {float(epoch_loss):.4f}')
            previous_best = epoch_loss
            checkpointer.save(is_best=True)

        else:
            logger.info(f'Loss in Epoch {epoch} has not improved from the previous best epoch')
            checkpointer.save(is_best=False)  # No save if save_best_only is True.


if __name__ == '__main__':

    # A hack to allow me maximum control over my training options.
    default_args = dict(
        batch_size=1,   # Batch size of 2 is enough to saturate my GPUs. 1 is almost enough but not quite.
        num_workers=2,
        init_lr=1E-3,
        gpu=1,  # Set to None for CPU mode.
        num_epochs=100,
        max_to_keep=5,
        verbose=False,
        save_best_only=True,
        train_root='data/multicoil_train',
        val_root='data/multicoil_val',
        max_imgs=0  # Maximum number of images to save.
        )

    my_args = create_arg_parser(**default_args).parse_args()

    main(my_args)
