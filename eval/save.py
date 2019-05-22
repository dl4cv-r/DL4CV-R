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


def make_val_loader(args):
    if args.single_coil:
        val_dir = '/home/veritas/PycharmProjects/DL4CV-R/data/singlecoil_val'
    else:
        val_dir = '/home/veritas/PycharmProjects/DL4CV-R/data/multicoil_val'

    dataset = SubmissionDataset(val_dir, transform=submission_slice_normalize_and_clip,
                                single_coil=args.single_coil, acc_fac=args.acc_fac, test_set=False, seed=9872)
    return DataLoader(dataset, args.batch_size, num_workers=args.num_workers)


def make_test_loader(args):
    if args.single_coil:
        test_dir = '/home/veritas/PycharmProjects/DL4CV-R/data/singlecoil_test'
    else:
        test_dir = '/home/veritas/PycharmProjects/DL4CV-R/data/multicoil_test'

    dataset = SubmissionDataset(test_dir, transform=submission_slice_normalize_and_clip,
                                single_coil=args.single_coil, acc_fac=args.acc_fac, test_set=True)
    return DataLoader(dataset, args.batch_size, num_workers=args.num_workers)


def restore_and_run_residual_model(args):
    from models.unet import UnetModel  # Might have to change model every time.
    torch.autograd.set_grad_enabled(False)  # Turns off gradient calculation.

    model = UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4).cuda(args.gpu)
    # Load model to gpu without optimizer and put in evaluation mode.
    model.load_state_dict(state_dict=torch.load(args.ckpt_path)['model_state_dict'])
    model.eval()

    if args.test_set:
        data_loader = make_test_loader(args)
    else:
        data_loader = make_val_loader(args)

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
        gpu=1,
        ckpt_path='/home/veritas/PycharmProjects/DL4CV-R/checkpoints/Trial 02  2019-04-23 18-40-17/ckpt_025.tar',
        data_root='/home/veritas/PycharmProjects/DL4CV-R/data/multicoil_test',
        test_set=True,
        single_coil=False,
        batch_size=12,
        num_workers=4,
        acc_fac=None,
        out_root='./submissions',
        out_name='residual_test_unet'
    )

    p_args = create_arg_parser(**defaults).parse_args()

    restore_and_run_residual_model(p_args)

# /home/veritas/PycharmProjects/DL4CV-R/eval/submissions/residual_unet_val_set --challenge multicoil
# MSE = 5.825e-11 +/- 1.824e-10 NMSE = 0.01228 +/- 0.01336 PSNR = 35.34 +/- 4.856 SSIM = 0.8742 +/- 0.124
# /home/veritas/PycharmProjects/DL4CV-R/eval/submissions/residual_unet_val_set_24 --challenge multicoil
# MSE = 5.82e-11 +/- 1.825e-10 NMSE = 0.01228 +/- 0.01343 PSNR = 35.35 +/- 4.866 SSIM = 0.8744 +/- 0.124

# python eval/evaluate.py --target-path /media/veritas/D/fastMRI/singlecoil_val
# --predictions-path /home/veritas/PycharmProjects/DL4CV-R/eval/submissions/single_val_trial --challenge singlecoil
# MSE = 3.229e-10 +/- 4.542e-10 NMSE = 0.1406 +/- 0.2271 PSNR = 26.35 +/- 8.352 SSIM = 0.5935 +/- 0.2834
