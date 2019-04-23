import torch
import h5py
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
import numpy as np

from tqdm import tqdm

from train.dataset import SubmissionDataset
from utils.args import create_arg_parser

# Data transforms may have to change depending on preprocessing.
from train.data_transforms import submission_slice_normalize_and_clip, recon_slice_normalize_and_clip


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
    print('Beginning saving')
    for file_name, recons in tqdm(reconstructions.items()):
        with h5py.File(out_dir / file_name, 'w') as f:
            f.create_dataset(name='reconstruction', data=recons, **gzip)
    else:
        print('Finished')


def restore_and_run_residual_model(args):
    from models.unet import UnetModel  # Might have to change model every time.
    torch.autograd.set_grad_enabled(False)  # Turns off gradient calculation.

    model = UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4).cuda(args.gpu)
    # Load model to gpu without optimizer and put in evaluation mode.
    model.load_state_dict(state_dict=torch.load(args.ckpt_path)['model_state_dict'])
    model.eval()

    dataset = SubmissionDataset(args.data_root, transform=submission_slice_normalize_and_clip,
                                single_coil=args.single_coil, acc_fac=args.acc_fac)

    data_loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)

    reconstructions = defaultdict(list)

    for ds_imgs, means, stds, acc_facs, fat_supps, file_dirs, slice_nums in tqdm(data_loader):
        shape = ds_imgs.shape[0]
        ds_imgs = ds_imgs.cuda(args.gpu).unsqueeze(dim=1)  # Makes the shape necessary for the model.
        pred_residuals = model(ds_imgs)  # The model is assumed to use residual learning
        recons = torch.squeeze(pred_residuals + ds_imgs).to('cpu')
        means = torch.squeeze(means).view((shape, 1, 1))
        stds = torch.squeeze(stds).view((shape, 1, 1))

        # This will work so long as batch size is not 320.
        recons = recon_slice_normalize_and_clip(data_slice=recons, mean=means, std=stds).numpy()

        for idx in range(shape):
            file_name = Path(file_dirs[idx]).name
            reconstructions[file_name].append((int(slice_nums[idx]), recons[idx]))

    recon_dict = dict()
    for file_name, slice_preds in reconstructions.items():
        # print('Reformatting: ', file_name)
        # for nn, pred in slice_preds:
        #     print(nn)
        #     print(pred.shape)
        recon_dict[file_name] = np.stack([pred for _, pred in sorted(slice_preds)], axis=0)

    print('Finished making recons')

    # Saving results.
    out_path = Path(args.out_root)
    out_path.mkdir(exist_ok=True)
    out_path = out_path / args.out_name
    out_path.mkdir(exist_ok=False)
    save_reconstructions(recon_dict, out_path)


if __name__ == '__main__':
    defaults = dict(
        gpu=0,
        ckpt_path='/home/veritas/PycharmProjects/DL4CV-R/checkpoints/Trial 01  2019-04-22 18-19-33/ckpt_027.tar',
        data_root='/home/veritas/PycharmProjects/DL4CV-R/data/multicoil_val',
        single_coil=False,
        batch_size=12,
        num_workers=4,
        acc_fac=None,  # This is because the validation set currently produces outputs for both 4-fold and 8-fold data
        out_root='./submissions',
        out_name='residual_unet_val_set_8'
    )

    # TODO: Fix the dataset system so that the validation set outputs just one of 4 or 8 fold acc.
    # Currently, both are being output and thus make outputs for both, making sorting and simple comparison impossible.
    # Also, training everything double time might be unfair in terms of reported epoch numbers etc.

    p_args = create_arg_parser(**defaults).parse_args()

    restore_and_run_residual_model(p_args)

