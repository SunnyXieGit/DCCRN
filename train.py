import wav_loader as loader
import net_config
import pickle
from torch.utils.data import DataLoader
import module as model_cov_bn
from si_snr import *
import train_utils
import os

########################################################################
dns_home = "D:\\33236\\PycharmProjects\\bishe\\DNS-Challenge\\datasets"
save_file = "./logs"
########################################################################

batch_size = 400  # calculate batch_size
load_batch = 100  # load batch_size(not calculate)
device = torch.device("cuda:0")  # device

lr = 0.01  # learning_rate
# load train and test name , train:test=4:1
if os.path.exists(r'./train_test_names.data'):
    train_test = pickle.load(open('./train_test_names.data', "rb"))
else:
    train_test = train_utils.get_train_test_name(dns_home)
train_noisy_names, train_clean_names, test_noisy_names, test_clean_names = \
    train_utils.get_all_names(train_test, dns_home=dns_home)

train_dataset = loader.WavDataset(train_noisy_names, train_clean_names, frame_dur=37.5)
test_dataset = loader.WavDataset(test_noisy_names, test_clean_names, frame_dur=37.5)
# dataloader将数据集train_dataset按照一定的批量大小（batch_size）和随机打乱（shuffle）的方式生成一个可迭代对象，用于在训练神经网络时批量加载数据。
train_dataloader = DataLoader(train_dataset, batch_size=load_batch, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=load_batch, shuffle=True)

dccrn = model_cov_bn.DCCRN_(
    n_fft=512, hop_len=int(6.25 * 16000 / 1000), net_params=net_config.get_net_params(), batch_size=batch_size,
    device=device, win_length=int((25 * 16000 / 1000))).to(device)

optimizer = torch.optim.Adam(dccrn.parameters(), lr=lr)
criterion = SiSnr()
train_utils.train(model=dccrn, optimizer=optimizer, criterion=criterion, train_iter=train_dataloader,
                  test_iter=test_dataloader, max_epoch=5, device=device, batch_size=batch_size, log_path=save_file,
                  just_test=False)
