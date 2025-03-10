
import torch

#计算信号与估计信号之间的尺度不变信噪比（SI-SNR）。
def si_snr(source, estimate_source, eps=1e-5):
    source = source.squeeze(1)
    estimate_source = estimate_source.squeeze(1)
    B, T = source.size()
    source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
    dot = torch.matmul(estimate_source, source.t())  # B , B
    s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
    e_noise = estimate_source - source
    snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
    lo = 0 - torch.mean(snr)
    return lo


class SiSnr(object):
    def __call__(self, source, estimate_source):
        return si_snr(source, estimate_source)


if __name__ == '__main__':
    # source = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]).view(2, 5).float()
    # estimate = torch.tensor([[1.5, 2.5, 3.5, 4.5, 5.5], [1.5, 2.5, 3.5, 4.5, 5.5]])
    # print(si_snr(source, estimate))
    source1 = torch.randn(1, 16000)  # 模拟音频信号
    estimate_source1 = torch.randn(1, 16000)
    loss1 = si_snr(source1, estimate_source1)
    print("1",loss1)  # 正常值应在 [-10, +20] 之间

    source = torch.randn(1, 16000)  # 模拟音频信号
    estimate_source = source.clone()
    loss = si_snr(source, estimate_source)
    print(loss)  # 理论值趋近于负无穷