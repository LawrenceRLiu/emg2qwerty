import torch
import torch.nn as nn
from typing import Tuple, Union, Optional


def temporal_loss(reconstructed_waveform:torch.FloatTensor,
                  original_waveform:torch.FloatTensor,
                  mask:Optional[Union[torch.FloatTensor, torch.BoolTensor]]=None,
                  p:Union[int, float]=2,
                  normalized:bool=True,
                  )->torch.FloatTensor:
    """Compute the temporal loss between the reconstructed and original signals, using the p-norm.
    If we have a mask, we will only consider the loss on the masked values. Reconstructed waveform may be 
    shorter than the original waveform, in which case the loss is only computed on the first N_samples_reconstructed
    samples of the original waveform.

    Args:
        reconstructed_waveform (torch.FloatTensor): the reconstructued waveform, of shape (..., N_samples_reconstructed)
        original_waveform (torch.FloatTensor): the original waveform, of shape (..., N_samples_original)
        mask (Optional[Union[torch.FloatTensor, torch.BoolTensor]], optional): the mask, only idxs that are true/1.0 are calculated. Defaults to None, if 
            None, all idxs are calculated. otherwise assumed to be of shape (..., N_samples_original)
        p (Union[int, float], optional): p of the norm. Defaults to 2.
        normalized (bool, optional): whether to normalize the loss by the norm of the original waveform. Defaults to True.
    Returns:
        torch.FloatTensor: loss of shape (1,)
    """

    errors = reconstructed_waveform - original_waveform[..., :reconstructed_waveform.shape[-1]]
    if mask is not None:
        errors = errors * mask[..., :reconstructed_waveform.shape[-1]]
    loss = (torch.abs(errors)**p).sum()

    if normalized:
        if mask is not None:
            loss = loss / torch.sum(torch.abs(reconstructed_waveform * mask[..., :reconstructed_waveform.shape[-1]])**p)
        else:
            loss = loss / torch.sum(torch.abs(reconstructed_waveform)**p)
    else:
        if mask is not None:
            loss = loss / mask.sum()
        else:
            loss = loss / reconstructed_waveform.numel()
    
    return loss

def spectral_loss(reconstructed_spectogram:torch.FloatTensor,
                  original_spectogram:torch.FloatTensor,
                  mask:Optional[Union[torch.FloatTensor, torch.BoolTensor]]=None,
                  p:Union[int, float]=2,
                  )->torch.FloatTensor:
    """Compute the spectral loss between the reconstructed and original signals, using the p-norm.
    If we have a mask, we will only consider the loss on the masked values. 
    can also handle patched spectrograms for ViT models.

    Args:
        reconstructed_spectogram (torch.FloatTensor): the reconstructued spectrum, of shape (..., N_fft//2 + 1, sequence_len//hop_length + 1) 
                    or of shape (..., n_patches, patch_numel)
        original_spectrum (torch.FloatTensor): the original spectrum, of shape (..., N_fft//2 + 1, sequence_len//hop_length + 1) 
                    or of shape (..., n_patches, patch_numel)
        mask (Optional[Union[torch.FloatTensor, torch.BoolTensor]], optional): the mask, only idxs that are true/1.0 are calculated. Defaults to None, if 
            None, all idxs are calculated. otherwise assumed to be of shape (..., N_fft//2 + 1, sequence_len//hop_length + 1) or (..., n_patches)
        p (Union[int, float], optional): p of the norm. Defaults to 2.

    Returns:
        torch.FloatTensor: loss of shape (1,)
    """
    errors = reconstructed_spectogram - original_spectogram
    if mask is not None:
        errors = (errors * mask) if len(mask.shape) == len(errors.shape) else (errors * mask.unsqueeze(-1))
    loss = (torch.abs(errors)**p if p != 2 else errors**2).sum()
    if mask is not None:
        loss = (loss / mask.sum()) if len(mask.shape) == len(errors.shape) else (loss / (mask.sum() * reconstructed_spectogram.shape[-1]))
    else:
        loss = loss / reconstructed_spectogram.numel()
    
    return loss
    
    
