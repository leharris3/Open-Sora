import torch.nn as nn

from .basics import avg_pool_nd, conv_nd


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=False, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or not sk:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if not sk:
            self.skep = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down:
            self.down_opt = nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.down:
            x = self.down_opt(x)
        if self.in_conv is not None:
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x

class BboxAdapter(nn.Module):
    def __init__(self, hidden_size, channels=[64, 128, 256, 512], nums_rb=2, cin=4, ksize=3, sk=False, use_conv=True, pretrained=False):
        super(BboxAdapter, self).__init__()
        self.hidden_size = hidden_size
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = nn.ModuleList()
        
        # Initial convolution to process input bbox ratios
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i-1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        
        # Calculate the final temporal dimension after downsampling
        self.T_final = channels[0] // (2 ** sum(1 for block in self.body if block.down))
        
        # Final layer to match the desired output dimension
        self.final_layer = nn.Conv2d(channels[-1], self.hidden_size, 1)  # Assuming 1152 is the desired output dimension

        if not pretrained:
            self.initialize_weights()

    def initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_weights)
        
        # Special initialization for the final layer
        nn.init.normal_(self.final_layer.weight, std=0.02)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x, T, B):
        # x shape: [B*T, 4]
        x = x.view(B, T, 4)
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, 4, T, 1]
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Process through ResNet blocks
        features = []
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)
        
        # Final processing
        x = self.final_layer(x)  # [B, 1152, T_final, 1]
        
        # Reshape to match the expected input of cross-attention
        x = x.squeeze(3).permute(0, 2, 1)  # [B, T_final, 1152]
        
        return x