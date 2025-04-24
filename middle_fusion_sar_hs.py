import torch
import torch.nn as nn

class Middle_fusion_en(nn.Module):

    def __init__(self,
                 conf_sar={'channels':[2,2], 'kernels':[3]},
                 conf_hs={'channels':[4,4], 'kernels':[3]},
                ):

        """
        conf_modality is a dict with:
            channels
            kernels

        e.g.
            conf_sar = {'channels':[2,32,64], 'kernels':[7,5]}
            conf_hs = {'channels':[4,32,64], 'kernels':[7,5]}
            conf_dem = {'channels':[1,32,64], 'kernels':[7,5]}
        """

        super(Middle_fusion_en, self).__init__()

        if(len(conf_sar['channels']) != len(conf_sar['kernels'])+1):
             raise Exception("sar configurations is wrong, channels length must be equal to kernels length + 1")
        if(len(conf_hs['channels']) != len(conf_hs['kernels'])+1):
            raise Exception("Hs configurations is wrong, channels length must be equal to kernels length + 1") 

        # sar convoltuions
        self.conv_sar = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=conf_sar['channels'][i], out_channels=conf_sar['channels'][i+1],
                          kernel_size=conf_sar['kernels'][i], padding=conf_sar['kernels'][i]//2, stride=1),
                nn.ReLU()) for i in range(len(conf_sar['kernels']))]
            )

        # hs convoltuions
        self.conv_hs = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=conf_hs['channels'][i], out_channels=conf_hs['channels'][i+1],
                          kernel_size=conf_hs['kernels'][i], padding=conf_hs['kernels'][i]//2, stride=1),
                nn.ReLU()) for i in range(len(conf_hs['kernels']))]
            )

    def forward(self, inp):
        sar, hs = inp

        sar = self.conv_sar(sar)
        hs = self.conv_hs(hs)

        return torch.cat((sar, hs), dim=1)


