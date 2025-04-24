import torch
import torch.nn as nn

class Middle_fusion_en(nn.Module):

    def __init__(self,
                 conf_rgb={'channels':[3,3], 'kernels':[3]},
                 conf_sar={'channels':[1,1], 'kernels':[3]}
                ):

        """
        conf_modality is a dict with:
            channels
            kernels

        e.g.
            conf_rgb = {'channels':[3,32,64], 'kernels':[7,5]}
            conf_hs = {'channels':[4,32,64], 'kernels':[7,5]}
            conf_sar = {'channels':[2,32,64], 'kernels':[7,5]}
        """

        super(Middle_fusion_en, self).__init__()

        if(len(conf_rgb['channels']) != len(conf_rgb['kernels'])+1):
             raise Exception("RGB configurations is wrong, channels length must be equal to kernels length + 1")
        if(len(conf_sar['channels']) != len(conf_sar['kernels'])+1):
            raise Exception("SAR configurations is wrong, channels length must be equal to kernels length + 1")

        # rgb convoltuions
        self.conv_rgb = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=conf_rgb['channels'][i], out_channels=conf_rgb['channels'][i+1],
                          kernel_size=conf_rgb['kernels'][i], padding=conf_rgb['kernels'][i]//2, stride=1),
                nn.ReLU()) for i in range(len(conf_rgb['kernels']))]
            )

        # sar convoltuions
        self.conv_sar = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=conf_sar['channels'][i], out_channels=conf_sar['channels'][i+1],
                          kernel_size=conf_sar['kernels'][i], padding=conf_sar['kernels'][i]//2, stride=1),
                nn.ReLU()) for i in range(len(conf_sar['kernels']))]
            )


    def forward(self, inp):
        rgb, sar = inp

        rgb = self.conv_rgb(rgb)
        sar = self.conv_sar(sar)

        return torch.cat((rgb, sar), dim=1)


