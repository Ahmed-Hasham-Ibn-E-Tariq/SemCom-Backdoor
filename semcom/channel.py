import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
def get_channel(channel_model):
    __chanel__ = {"AWGN": AWGN_Channel,"Rayleigh":Rayleigh}
    return __chanel__[channel_model]





class Channel(nn.Module):
    """
    Channel model supporting AWGN and Rayleigh fading, with batch-wise SNR.
    """

    def __init__(self, channel_type):
        super(Channel, self).__init__() # corrected here
        self.chan_type = channel_type

    def gaussian_noise_layer(self, input_layer, std):
        """
        生成复高斯噪声，确保 SNR 计算正确
        """
        device = input_layer.device
        batch_size = input_layer.shape[0]

        # 生成独立的实部和虚部高斯噪声
        noise_real = torch.randn_like(input_layer)
        noise_imag = torch.randn_like(input_layer)

        # 将 std 扩展到 batch 维度
        std_expanded = std.view(batch_size, *([1] * (input_layer.dim() - 1))).to(device) # corrected here, move to device

        # 按标准差缩放
        noise_real = noise_real * std_expanded
        noise_imag = noise_imag * std_expanded

        # 复数噪声
        noise = noise_real + 1j * noise_imag

        return input_layer + noise.to(device)

    def rayleigh_noise_layer(self, input_layer, std):
        """
        生成瑞利衰落 + 复高斯噪声，确保 SNR 计算正确
        """
        device = input_layer.device
        batch_size = input_layer.shape[0]

        # 生成复高斯噪声
        noise_real = torch.randn_like(input_layer)
        noise_imag = torch.randn_like(input_layer)

        # 生成瑞利衰落信道 h
        h_real = torch.randn_like(input_layer)
        h_imag = torch.randn_like(input_layer)
        h = torch.sqrt(h_real ** 2 + h_imag ** 2) / np.sqrt(2)  # 瑞利分布归一化

        # 将 std 扩展到 batch 维度
        std_expanded = std.view(batch_size, *([1] * (input_layer.dim() - 1))).to(device) # corrected here, move to device

        # 计算噪声
        noise_real = noise_real * std_expanded
        noise_imag = noise_imag * std_expanded
        noise = noise_real + 1j * noise_imag

        return input_layer * h.to(device) + noise.to(device), h


    def complex_normalize(self, x, power=1):
        """
        ... (文档字符串保持不变) ...
        """
        batch_size = x.shape[0]

        # 计算每个样本的功率 (修正为复数功率)
        pwr = torch.mean(torch.abs(x)**2, dim=tuple(range(1, x.dim())), keepdim=True) * 2  # 使用 torch.abs(x)**2 计算复数功率

        # 避免除零错误
        pwr = torch.clamp(pwr, min=1e-9)

        # 归一化每个样本
        out = np.sqrt(power) * x / torch.sqrt(pwr)

        return out, pwr

    def forward(self, input, chan_param, avg_pwr=False):
        batch_size = input.shape[0]

        # 每个样本单独归一化
        channel_tx, pwr = self.complex_normalize(input, power=1) # pwr is batch_size x 1 x 1...

        input_shape = channel_tx.shape
        channel_in = channel_tx.view(batch_size, -1)
        L = channel_in.shape[1]
        channel_in = channel_in[:, :L // 2] + channel_in[:, L // 2:] * 1j

        # 通过信道
        if self.chan_type=='rayleigh':
            channel_output, h = self.complex_forward(channel_in, chan_param)
        else:
            channel_output = self.complex_forward(channel_in, chan_param)

        # 复数分量拆分回 real 和 imag
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)], dim=1)
        channel_output = channel_output.view(input_shape)

        # 归一化，使得功率匹配
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise # important, adding noise to the transmitted signal not output

            return channel_tx, None, pwr # removed normalization by pwr, channel_tx is already normalized to power 1

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            return channel_output, h, pwr # removed normalization by pwr, channel_output is already processed based on power 1 input
        
    
    def forward_with_channelEQ(self, input, chan_param, avg_pwr=False):
        batch_size = input.shape[0]

        # 每个样本单独归一化
        channel_tx, pwr = self.complex_normalize(input, power=1) # pwr is batch_size x 1 x 1...

        input_shape = channel_tx.shape
        channel_in = channel_tx.view(batch_size, -1)
        L = channel_in.shape[1]
        channel_in = channel_in[:, :L // 2] + channel_in[:, L // 2:] * 1j

        # 通过信道
        if self.chan_type=='rayleigh':
            channel_output, h = self.complex_forward(channel_in, chan_param)
            # 获取噪声功率
            device = channel_output.device
            chan_param_expanded = chan_param.view(batch_size, *([1] * (h.dim() - 1))).to(device)
            sigma_n = 1.0 / (10 ** (chan_param_expanded / 10))
            h_conj = torch.conj(h)
            h_abs_squared = torch.abs(h)**2
            
            # 根据h和SNR自动确定是使用ZF还是MMSE
            snr_threshold = 30  # dB阈值
            channel_quality = 10 * torch.log10(h_abs_squared + 1e-10)
            effective_snr = channel_quality + chan_param_expanded
            
            # 创建掩码：高SNR用ZF，低SNR用MMSE
            zf_mask = (effective_snr > snr_threshold).float()
            
            # 计算两种系数
            zf_coef = h_conj / (h_abs_squared + 1e-10)
            mmse_coef = h_conj / (h_abs_squared + sigma_n) 
            
            # 融合系数
            eq_coef = zf_mask * zf_coef + (1 - zf_mask) * mmse_coef
            channel_output = channel_output * eq_coef   
            power_before_eq = torch.mean(torch.abs(channel_output/eq_coef)**2, dim=1, keepdim=True)
            power_after_eq = torch.mean(torch.abs(channel_output)**2, dim=1, keepdim=True)
            power_ratio = power_after_eq / (power_before_eq + 1e-10)
            
            # 检测异常均衡 (功率放大过大)
            abnormal_mask = (power_ratio > 5.0).float()
            
            # 对异常区域应用限制性均衡
            if abnormal_mask.sum() > 0:
                # 对信号进行功率限制
                scale_factor = 5.0 / (power_ratio + 1e-10)
                scale_factor = torch.where(abnormal_mask > 0, scale_factor, torch.ones_like(scale_factor))
                channel_output = channel_output * scale_factor.expand_as(channel_output)# 在INN反向传播前添加，位于train.py中
        else:
            channel_output = self.complex_forward(channel_in, chan_param)

        # 复数分量拆分回 real 和 imag
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)], dim=1)
        channel_output = channel_output.view(input_shape)

        # 归一化，使得功率匹配
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise # important, adding noise to the transmitted signal not output

            return channel_tx, None, pwr # removed normalization by pwr, channel_tx is already normalized to power 1

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            return channel_output, h, pwr # removed normalization by pwr, channel_output is already processed based on power 1 input


    def complex_forward(self, channel_in, chan_param):
        """
        处理信道传播和噪声，确保 SNR 计算正确
        """
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            # 计算 SNR 对应的标准差
            sigma = torch.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))  # 符号功率归一化时的噪声计算

            # 生成加性高斯噪声
            return self.gaussian_noise_layer(channel_in, std=sigma)

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            # 计算 SNR 对应的标准差
            sigma = torch.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            
            # 生成瑞利衰落 + 噪声
            return self.rayleigh_noise_layer(channel_in, std=sigma)



    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx









