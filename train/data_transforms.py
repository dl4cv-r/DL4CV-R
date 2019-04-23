import numpy as np


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
        or numpy.ndarray: normalized array
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor or numpy.ndarray): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
            or numpy.ndarray: normalized array
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


def crop_slice(img_slice, view_size=320):
    top = (img_slice.shape[-2] - view_size) // 2
    left = (img_slice.shape[-1] - view_size) // 2
    return img_slice[..., top:top+view_size, left:left+view_size]


def slice_normalize_and_clip(ds_slice, gt_slice, fat_supp, eps=0.):
    """
    Inputs are expected to be numpy arrays, not Pytorch tensors.
    I leave conversion for later.
    This attempts to be the same pre-processing that the Facebook team used.
    """
    ds_slice = crop_slice(ds_slice)
    ds_slice, mean, std = normalize_instance(ds_slice, eps)
    gt_slice = normalize(gt_slice, mean, std, eps)
    data = np.clip(ds_slice, a_min=-6, a_max=6)
    labels = np.clip(gt_slice, a_min=-6, a_max=6)
    return data, labels


def submission_slice_normalize_and_clip(ds_slice, fat_supp, eps=0.):  # For submission
    """
    Inputs are expected to be numpy arrays, not Pytorch tensors.
    I leave conversion for later.
    This attempts to be the same pre-processing that the Facebook team used.
    """
    ds_slice = crop_slice(ds_slice)
    ds_slice, mean, std = normalize_instance(ds_slice, eps)
    data = np.clip(ds_slice, a_min=-6, a_max=6)
    return data, mean, std


def recon_slice_normalize_and_clip(data_slice, mean, std):
    return data_slice * std + mean
