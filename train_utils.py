import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
#from pesq import pesq
import os
import gc
import sys
import time

#根据训练测试划分和DNS数据集路径获取文件名。
def get_all_names(train_test, dns_home):
    train_names = train_test["train"]
    test_names = train_test["test"]

    train_noisy_names = []
    train_clean_names = []
    test_noisy_names = []
    test_clean_names = []

    for name in train_names:
        code = str(name).split('_')[-1]
        clean_file = os.path.join(dns_home, 'clean_output', 'clean_fileid_%s' % code)
        noisy_file = os.path.join(dns_home, 'noisy', name)
        train_clean_names.append(clean_file)
        train_noisy_names.append(noisy_file)
    for name in test_names:
        code = str(name).split('_')[-1]
        clean_file = os.path.join(dns_home, 'clean_output', 'clean_fileid_%s' % code)
        noisy_file = os.path.join(dns_home, 'noisy', name)
        test_clean_names.append(clean_file)
        test_noisy_names.append(noisy_file)
    return train_noisy_names, train_clean_names, test_noisy_names, test_clean_names

#在测试集上评估模型。
def test_epoch(model, test_iter, device, criterion, batch_size, test_all=False):
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        i = 0
        for ind, (x, y) in enumerate(test_iter):
            x = x.view(x.size(0) * x.size(1), x.size(2)).to(device).float()
            y = y.view(y.size(0) * y.size(1), y.size(2)).to(device).float()
            for index in range(0, x.size(0) - batch_size, batch_size):
                x_item = x[index:index + batch_size, :].squeeze(0)
                y_item = y[index:index + batch_size, :].squeeze(0)
                y_p = model(x_item, train=False)
                loss = criterion(source=y_item.unsqueeze(1), estimate_source=y_p)
                loss_sum += loss.item()
                i += 1
            if not test_all:
                break
    return loss_sum / i

#训练模型并记录损失。
def train(model, optimizer, criterion, train_iter, test_iter, max_epoch, device, batch_size, log_path, just_test=False):
    train_losses = []
    test_losses = []
    for epoch in range(max_epoch):
        loss_sum = 0
        i = 0
        for step, (x, y) in enumerate(train_iter):
            if epoch == 0 and step == 0:  # 仅检查第一个 batch
                print("[Data Check] Raw input x shape:", x.shape)
                print("[Data Check] Raw label y shape:", y.shape)
                # 打印第一个样本的部分数据
                print("First sample of x (first 15 values):", x[0, 0, :15])
                print("First sample of y (first 15 values):", y[0, 0, :15])
            x = x.view(x.size(0) * x.size(1), x.size(2)).to(device).float()
            y = y.view(y.size(0) * y.size(1), y.size(2)).to(device).float()
            if epoch == 0 and step == 0:  # 仅检查第一个 batch
                print("Reshaped x shape:", x.shape)
                print("Reshaped y shape:", y.shape)
                # 检查是否 x 和 y 不同
                if torch.allclose(x, y, atol=1e-4):
                    print("⚠️ 警告：输入 x 和标签 y 完全相同！存在数据泄露！")
                else:
                    print("输入 x 和标签 y 不同，数据正常。")
            shuffle = torch.randperm(x.size(0))
            x = x[shuffle]
            y = y[shuffle]
            # 计算余弦相似度（仅检查第一个样本）
            if epoch == 0 and step == 0:
                print("shuffled x shape:", x.shape)
                print("shuffled y shape:", y.shape)
                # 检查是否 x 和 y 不同
                if torch.allclose(x, y, atol=1e-4):
                    print("⚠️ 警告：输入 x 和标签 y 完全相同！存在数据泄露！")
                else:
                    print("输入 x 和标签 y 不同，数据正常。")
                x_sample = x[0].view(1, -1)  # [1, T]
                y_sample = y[0].view(1, -1)  # [1, T]
                cosine_sim = torch.nn.functional.cosine_similarity(x_sample, y_sample, dim=1).item()
                print(f"输入 x 和标签 y 的余弦相似度：{cosine_sim:.4f}")
                if abs(cosine_sim) > 0.8:
                    print("⚠️ 警告：输入和标签高度相似，可能存在数据泄露！")
            for index in range(0, x.size(0) - batch_size + 1, batch_size):
                model.train()
                x_item = x[index:index + batch_size, :].squeeze(0)
                y_item = y[index:index + batch_size, :].squeeze(0)
                optimizer.zero_grad()
                y_p = model(x_item)
                #criterion = SiSnr()
                loss = criterion(source=y_item.unsqueeze(1), estimate_source=y_p)
                if step == 0 and index == 0 and epoch == 0:
                    loss.backward()
                    loss_sum += loss.item()
                    i += 1
                    test_loss = test_epoch(model, test_iter, device, criterion, batch_size=batch_size, test_all=False)
                    print(
                        "first test step:%d,ind:%d,train loss:%.5f,test loss:%.5f" % (
                            step, index, loss_sum / i, test_loss)
                    )
                    train_losses.append(loss_sum / i)
                    test_losses.append(test_loss)
                else:
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    i += 1
            if step % int(len(train_iter) // 10) == 0 or step == len(train_iter) - 1:
                test_loss = test_epoch(model, test_iter, device, criterion, batch_size=batch_size, test_all=False)
                print(
                    "epoch:%d,step:%d,train loss:%.5f,test loss:%.5f,time:%s" % (
                        epoch, step, loss_sum / i, test_loss, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
                    )
                )
                train_losses.append(loss_sum / i)
                test_losses.append(test_loss)
                plt.plot(train_losses)
                plt.plot(test_losses)
                plt.savefig(os.path.join(log_path, "loss_time%s_epoch%d_step%d.png" % (
                    time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), epoch, step)), dpi=150)
                plt.show()
            if (step % int(len(train_iter) // 3) == 0 and step != 0) or step == len(train_iter) - 1:
                print("save model,epoch:%d,step:%d,time:%s" % (
                    epoch, step, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())))
                torch.save(model, os.path.join(log_path, "parameter_epoch%d_%s.pth" % (
                    epoch, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))))
                pickle.dump({"train loss": train_losses, "test loss": test_losses},
                            open(os.path.join(log_path, "loss_time%s_epoch%d.log" % (
                                time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), epoch)), "wb"))
            if just_test:
                break

#划分训练集和测试集并保存文件名。
def get_train_test_name(dns_home):
    all_name = []
    for i in os.walk(os.path.join(dns_home, "noisy")):
        for name in i[2]:
            all_name.append(name)
    train_names = all_name[:-len(all_name) // 5]
    test_names = all_name[-len(all_name) // 5:]
    print(len(train_names))
    print(len(test_names))
    data = {"train": train_names, "test": test_names}
    pickle.dump(data, open("./train_test_names.data", "wb"))
    return data
