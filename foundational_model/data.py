import emg2qwerty.data
from dataclasses import dataclass

@dataclass
class PretrainSingleChannel_emg2qwerty(emg2qwerty.data.WindowedEMGDataset):

    n_channels = 32 #hardcoded because I didn't care to get this from the config
    def __len__(self):
        if hasattr(self, "n_channels"):

            return super().__len__()*self.n_channels
        else:
            return super().__len__()
    def __getitem__(self, idx):

        emg,*_ = super().__getitem__(idx//self.n_channels)
        print(emg.shape)
        i = idx%self.n_channels
        hand = i % 2
        channel = i // 2
        data = emg[:, hand, channel
                   ]
        return {"emg":data}
    

