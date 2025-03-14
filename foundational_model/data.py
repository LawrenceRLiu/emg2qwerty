import emg2qwerty.data
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch


# Some thing I noticed was that for the first index, the output shape will be different because of padding for the windowed dataset, so padding should be set to zero unless there is a bug

@dataclass
class PretrainChannelWise_emg2qwerty(emg2qwerty.data.WindowedEMGDataset):

    n_channels = 32 #hardcoded because I didn't care to get this from the config
    def __len__(self):
        return super().__len__()*self.n_channels
    
    def __getitem__(self, idx) -> Dict[str,torch.Tensor]:
        """get item from the dataset

        Returns:
            output: Dict[str:torch.Tensor]: a dictionary containing the emg data, of shape (1, window_size)
            There is a one so that we unify the shape of the output between different methods of getting the data
        """

        emg,*_ = super().__getitem__(idx//self.n_channels) #shape of (window_size, 2, 16)
        i = idx%self.n_channels
        hand = i % 2
        channel = i // 2
        data = emg[:, hand, channel]
        return {"emg":data.unsqueeze(0)}
    


@dataclass
class PretrainHandWise_emg2qwerty(emg2qwerty.data.WindowedEMGDataset):

    n_hands = 2 #2 hands, I hope!
    def __len__(self):
        return super().__len__()*self.n_hands
    def __getitem__(self, idx)-> Dict[str,torch.Tensor]:
        """get item from the dataset

        Returns:
            output: Dict[str:torch.Tensor]: a dictionary containing the emg data, of shape (16, window_size)
        """

        emg,*_ = super().__getitem__(idx//self.n_hands) #shape of (window_size, 2, 16)
        hand = idx % self.n_hands
        data = emg[:, hand, :].T
        return {"emg":data}


@dataclass
class Pretrain_emg2qwerty(emg2qwerty.data.WindowedEMGDataset):
    flatten: bool = True #whether to flatten the data

    def __getitem__(self, idx)-> Dict[str,torch.Tensor]:
        """get item from the dataset

        Returns:
            output: Dict[str:torch.Tensor]: a dictionary containing the emg data, of shape (16, 2, window_size) if flatten is False, else (16*2, window_size)
        """

        emg,*_ = super().__getitem__(idx)
        if self.flatten:
            data = emg.reshape(-1, emg.shape[-1])
        else:
            data = emg
        return {"emg":data}
    