from torch import nn
from utils.config import scale_factor

# @class    : SUPERCNN
# @brief    : It is a CNN network. It increases the resolution to the scale factor, which is determined as output, 
#             by passing the given image through the layers created sequentially.
# @param    : None

class SUPRESCNN(nn.Module):
    def __init__(self):
        super(SUPRESCNN, self).__init__()
        
        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=56, kernel_size=5, padding=2),
            nn.PReLU(56) )
        
        self.mid_part1 = nn.Sequential(
            nn.Conv2d(in_channels=56,out_channels=12, kernel_size=1, padding=0),
            nn.PReLU(12))
        
        self.mid_part2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.PReLU(12))
        
        self.mid_part3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=56, kernel_size=1, padding=0),
            nn.PReLU(56))
        
        self.last_part = nn.ConvTranspose2d(in_channels=56, out_channels=1, kernel_size=9, stride = scale_factor, padding=4, output_padding=scale_factor-1)
        
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part1(x)
        x = self.mid_part2(x)
        x = self.mid_part3(x)
        x = self.last_part(x)
        
        return x