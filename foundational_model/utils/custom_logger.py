# a custom logger that logs built off the 
# wandb torch lightning logger that will save the spectograms
# and the waveforms as plots

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
from wand.image import Image
from torch import nn
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from typing import List, Tuple, Union, Dict

class CustomPretrainLogger(WandbLogger):
    cache: Dict[str, plt.Figure] = {}
    
    def log_plot_to_wandb(self,
                        fig: plt.Figure,
                        name: str,
                        close: bool = True):
        """Logs a matplotlib figure to wandb
        Args:
            fig (plt.Figure): the figure to log
            name (str): the name of the figure
            close (bool, optional): whether to close the figure after logging. Defaults to True.
        """
        # buf = BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        wandb.log({name: wandb.Image(fig)})
        if close:
            plt.close(fig)

    def dump_cache(self, prefix: str = "",
                   save_to_disk: bool = False,
                   dump_matched_only: bool = False,
                   matching_substr: str = ""):
        """Dumps the cache to wandb and optionally to disk
        Args:
            prefix (str, optional): the prefix to use for the cache. Defaults to "".
            save_to_disk (bool, optional): whether to save the cache to disk. Defaults to False.
            dump_matched_only (bool, optional): whether to only dump the plots that have a matching prefix. Defaults to False.
            matching_prefix (str, optional): the sub string to match. Defaults to "".
        """

        prefix_use = f"{self.save_dir}/{wandb.run.name}/{prefix}"
        print(f"dumping cache to {prefix_use}")
        os.makedirs(prefix_use, exist_ok=True)
        cache_keys = list(self.cache.keys())
        for name in cache_keys:
            fig = self.cache[name]
            if dump_matched_only and not matching_substr in name:
                continue
            self.log_plot_to_wandb(fig, f"{prefix}/{name}", close = not save_to_disk)

            if save_to_disk:

                fig.savefig(f"{prefix_use}/{name}.png")
                plt.close(fig)
            
            del self.cache[name]
        
        print("remaining cache", len(self.cache))

    def log_spectograms(self,
                        spectograms: List[torch.FloatTensor],
                        names: List[str],
                        spec_size:int = 5,
                        layout: Union[Tuple[int, int], None] = None,
                        plot_name: str = "spectograms"):
        """Logs a list of spectograms to wandb
        Args:
            spectograms (List[torch.FloatTensor]): the list of spectograms to log
            names (List[str]): the names of the spectograms
            spec_size (int, optional): the size of the spectograms. Defaults to 5.
            layout (Union[Tuple[int, int], None], optional): the layout of the spectograms. Defaults to None.
        """
        #if the layout is none, just log a flat list
        if layout is None:
            layout = (1, len(spectograms))

        fig, axs = plt.subplots(*layout, figsize=( spec_size * layout[1], spec_size * layout[0]))
        #unsqueeze the axs if it is a single axis
        if len(axs.shape) == 1:
            axs = np.expand_dims(axs, 0)
        for i, (spec, name) in enumerate(zip(spectograms, names)):
            ax = axs[i%layout[0], i//layout[0]]
            ax.imshow(spec.squeeze().cpu().numpy(), origin = "lower")
            ax.set_title(name)
            # ax.axis('off')
        
        self.cache[plot_name] = fig

    def log_waveforms(self, 
                        waveforms: List[torch.FloatTensor],
                        names: List[str],
                        plot_name: str = "waveforms",
                        seperate_plots: bool = False):
        """Logs a list of waveforms to wandb
        Args:
            waveforms (List[torch.FloatTensor]): the list of waveforms to log
            names (List[str]): the names of the waveforms
            plot_name (str, optional): the name of the plot. Defaults to "waveforms".
            seperate_plots (bool, optional): whether to log the waveforms in seperate subplots. Defaults to False.
        """
        #if we are logging in seperate plots, we will log each waveform in a seperate subplots one on top of the other
        if seperate_plots:
            fig, axs = plt.subplots(len(waveforms), 1, figsize=(5, 2*len(waveforms)), sharex=True, sharey=True)
            for waveform, name, ax in zip(waveforms, names, axs):
                ax.plot(waveform.squeeze().cpu().numpy())
                ax.set_title(name)
                # ax.axis('off')
        else:
            fig = plt.figure(figsize = (5,2))
            for waveform, name in zip(waveforms, names):
                plt.plot(waveform.squeeze().cpu().numpy(), label = name)
            plt.legend()
            # plt.axis('off')
        
        self.cache[plot_name] = fig
        
    
        
        