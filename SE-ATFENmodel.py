'''
2025.1.12
by kx
'''




import numpy as np

import torch
import torch.nn as nn
from kymatio.scattering1d.frontend.base_frontend import ScatteringBase1D
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import init
class ModulusStable(Function):

    @staticmethod
    def forward(ctx, x):

        ctx.p = 2
        ctx.dim = -1
        ctx.keepdim = False

        output = (x[...,0] * x[...,0] + x[...,1] * x[...,1]).sqrt()

        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_tensors

        if ctx.dim is not None and ctx.keepdim is False and x.dim() != 1:
            grad_output = grad_output.unsqueeze(ctx.dim)
            output = output.unsqueeze(ctx.dim)

        grad_input = x.mul(grad_output).div(output)

        # Special case at 0 where we return a subgradient containing 0
        grad_input.masked_fill_(output == 0, 0)

        return grad_input
def cdgmm(A, B):


    assert A.shape[-1] == 2, "A should have a last dimension of size 2 (real and imaginary parts)."
    assert B.shape[-1] in [1, 2], "B should have a last dimension of size 1 (real) or 2 (complex)."


    if B.shape[-1] == 1:
        return A * B

    else:

        A_real, A_imag = A[..., 0], A[..., 1]
        B_real, B_imag = B[..., 0], B[..., 1]


        C_real = A_real * B_real - A_imag * B_imag
        C_imag = A_real * B_imag + A_imag * B_real


        C = torch.stack([C_real, C_imag], dim=-1)

        return C



class LearnableSigmoid(nn.Module):

    def __init__(self, init_shift=1.0, init_base=10.0):
        super(LearnableSigmoid, self).__init__()
        self.shift = nn.Parameter(torch.tensor(init_shift))
        self.base = nn.Parameter(torch.tensor(init_base))

    def forward(self, x):
        return 1 / (1 + torch.pow(self.base, -(x - self.shift)))

class SE_attenton_block(nn.Module):
    def __init__(self, in_channels=None,  input_size=113, reduction=4):

        super(SE_attenton_block, self).__init__()
        self.linear = nn.Linear(input_size, 2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):

                init.constant_(m.weight, val=1)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):  # (batch_size, n_filters, signal_length)
        batch_size, channel, signal_length = x.shape

        result = torch.hann_window(signal_length) * x
        fft_result = torch.abs(torch.fft.rfft(result)) / signal_length

        out_acti = self.linear(fft_result)
        out_acti = self.relu(out_acti)

        weights = self.linear.weight.detach().numpy()
        bases = self.linear.bias.detach().numpy()
        np_out = out_acti.detach().numpy()
        w = out_acti[..., 0].view(batch_size, channel, 1)

        return w

class wavelet_scat_embeded_block3(nn.Module, ScatteringBase1D):

    def __init__(self, J=10, shape=4096, Q=11, T=None, max_order=2, average=None,
                 oversampling=0, out_type='array', backend='torch'):
        nn.Module.__init__(self)
        self.frontend_name = 'torch'
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                                  oversampling, out_type, backend)
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        self.meta = ScatteringBase1D.meta(self)
        self.register_filters()
        buffer_dict = dict(self.named_buffers())
        self.psi1_matrix = torch.stack([buffer_dict['tensor' + str(n)] for n in range(len(buffer_dict))],
                                       dim=0)

    def register_filters(self):

        n = 0

        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(
                    psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1


    def forward(self, x):  # x->(batch, 1, n_samples)

        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)

        U_0 = F.pad(x, (self.pad_left, self.pad_right), mode='reflect')
        U_0 = U_0[..., None]
        x_r = torch.zeros(U_0.shape[:-1] + (2,), dtype=x.dtype, layout=x.layout, device=x.device)
        x_r[..., 0] = U_0[..., 0]
        U_0_hat = torch.view_as_real(torch.fft.fft(torch.view_as_complex(x_r)))

        U_1_c = cdgmm(U_0_hat, self.psi1_matrix)

        U_1_c = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(U_1_c)))
        U_1_m = ModulusStable.apply(U_1_c)
        first_scat_features = U_1_m[..., self.ind_start[0]:self.ind_end[0]]

        order1 = np.where(self.meta['order'] == 1)
        sigma_order1 = [self.meta['sigma'][sigma_index, 0] for sigma_index in order1[0]]

        norm_factor = torch.from_numpy(np.log2(np.array(sigma_order1) / sigma_order1[-1] + 0.2) / np.log(1.2)).to(torch.float32)

        first_scat_features_norm = first_scat_features / norm_factor[:, np.newaxis]

        first_scat_features_x = F.interpolate(first_scat_features.unsqueeze(1), size=(224, 224), mode='bilinear',
                                              align_corners=False).squeeze(1)

        first_scat_features_norm = F.interpolate(first_scat_features_norm.unsqueeze(1), size=(224, 224), mode='bilinear',
                                            align_corners=False).squeeze(1)
        first_scat_features = first_scat_features_norm - first_scat_features_norm.mean(dim=2, keepdim=True)

        return first_scat_features_x, first_scat_features


class residualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        torch.manual_seed(42)

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.main_branch(x)
        shortcut = self.shortcut(x)
        return F.relu(residual + shortcut)



class ResidualBlock2D(nn.Module):
    def __init__(self, num_class=3):
        super(ResidualBlock2D, self).__init__()
        torch.manual_seed(42)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            # nn.AdaptiveMaxPool2d(64)
        )

        self.layer1 = self._make_layer(16, 32, num_blocks=1, stride=2)
        self.layer2 = self._make_layer(32, 64, num_blocks=1, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=1, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self._initialize_weights()  # 自适应平均池化
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [
            residualBlock2D(in_channels, out_channels, stride)
        ]
        for _ in range(1, num_blocks):
            layers.append(residualBlock2D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.initial_conv(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # ( batch, 128,8,8)

        x = self.global_pool(x)
        x = x.view(batch_size, -1)

        return x


class Wavelet_SE_attention_Net8(nn.Module):

    def __init__(self, num_classes):
        super(Wavelet_SE_attention_Net8, self).__init__()
        torch.manual_seed(42)

        self.wave_scat = wavelet_scat_embeded_block3(J=10, shape=4096, Q=11, T=0, max_order=2, average=False,
                                                    oversampling=0, out_type='array', backend='torch')  # 散射子模块的实例化

        self.SE_attention_threshold = SE_attenton_block(in_channels=224, reduction=4)


        self.feature_extractor = ResidualBlock2D()

        self.fc = nn.Linear(128, num_classes)


    def predict(self, x):

        wx, x = self.wave_scat(x)

        w, threshold = self.SE_attention_threshold(
            x)

        branch_attention = wx * w.expand_as(x)



        new_x = branch_attention

        x_e = self.feature_extractor(new_x)
        # x_e = self.feature_extractor(x)

        y = self.fc(x_e)
        return y

    def forward(self, x):

        wx, x = self.wave_scat(x)

        w, threshold = self.SE_attention_threshold(x)

        branch_attention = wx * w.expand_as(x)

        new_x = branch_attention

        x_e = self.feature_extractor(new_x)


        y = self.fc(x_e)
        return y, w, x_e


# 模型测试
if __name__ == '__main__':

    model = Wavelet_SE_attention_Net8(3)


    print(model)
    model_conv = nn.Sequential(*list(model.children())[:-1])


    print(model_conv)

    print(1)



