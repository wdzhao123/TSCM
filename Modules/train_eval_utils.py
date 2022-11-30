import sys
from torch.autograd import Variable
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from apex import amp


def train_one_epoch(model, optimizer, data_loader, epoch, device='cuda'):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, (img, label) in enumerate(data_loader):
        img, label = Variable(img).cuda(), Variable(label).cuda()
        pred = model(img)

        loss = loss_function(pred, label.squeeze())
        loss.backward()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


def train_one_epoch_IN(model, optimizer, data_loader, epoch, alpha=0.0, beta=0.0, gamma=0.0, device='cuda'):
    # Cn_Sm indicates feature combined by Content from n & Style from m

    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    mean_CE_loss = torch.zeros(1).to(device)
    mean_cos_C2S1_C3S1_loss = torch.zeros(1).to(device)
    mean_cos_C1S2_C2S1_loss = torch.zeros(1).to(device)
    mean_cos_C1S2_C1S3_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    non_times = 0
    for step, (img, label) in enumerate(data_loader):
        img, label = Variable(img).cuda(), Variable(label).cuda()
        batch_size = img.size()[0]  # N the batch size
        img_tuple = torch.split(img, 3, dim=1)  # img(N*3C*H*W) ---> img_tuple(3*(N*C*H*W))

        img = torch.cat(img_tuple, dim=0)  # img_tuple(3*(N*C*H*W)) ---> img(3N*C*H*W)
        pred = model(img)  # classification output(5N*15)

        pred_tuple = torch.split(pred, batch_size,
                                 dim=0)  # classification output(3N*15) ---> classification output(3*(N*15))
        pred_1 = pred_tuple[0]  # main img out, class1(N*15)
        C1_S2 = pred_tuple[1]  # (N*15)
        C1_S3 = pred_tuple[2]  # (N*15)
        C2_S1 = pred_tuple[3]  # (N*15)
        C3_S1 = pred_tuple[4]  # (N*15)

        CE_loss = loss_function(pred_1, label.squeeze())  # Cross Entropy Loss
        cosine_dis_C1S2_C1S3 = -torch.log(
            (F.cosine_similarity(C1_S2, C1_S3, dim=1).sum() / (batch_size + 1e-8) + 1) / 2)  # Intra-class Compactness
        cosine_dis_C1S2_C2S1 = -torch.log(
            (F.cosine_similarity(C1_S2, C2_S1, dim=1).sum() / (batch_size + 1e-8) + 1) / 2)  # Intra-class Interaction

        cos_loss = nn.CosineEmbeddingLoss()
        cosine_dis_C2S1_C3S1 = -torch.log(
            torch.tensor([1.0]).cuda() - cos_loss(C2_S1, C3_S1, torch.tensor([-1.]).cuda()))  # Inter-class Dispersion

        loss = CE_loss \
               + alpha * cosine_dis_C2S1_C3S1 \
               + beta * cosine_dis_C1S2_C2S1 \
               + gamma * cosine_dis_C1S2_C1S3
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        mean_CE_loss = (mean_CE_loss * step + CE_loss.detach()) / (step + 1)  # update mean losses
        mean_cos_C2S1_C3S1_loss = (mean_cos_C2S1_C3S1_loss * step + cosine_dis_C2S1_C3S1.detach()) / (step + 1)
        mean_cos_C1S2_C2S1_loss = (mean_cos_C1S2_C2S1_loss * step + cosine_dis_C1S2_C2S1.detach()) / (step + 1)
        mean_cos_C1S2_C1S3_loss = (mean_cos_C1S2_C1S3_loss * step + cosine_dis_C1S2_C1S3.detach()) / (step + 1)

        data_loader.desc = "[epoch {}] mean loss {},\n " \
                           "CrossEntropy {}, " \
                           "Inter-class Dispersion {}, " \
                           "Intra-class Compactness {}, " \
                           "Intra-class Interaction {}. ". \
            format(epoch,
                   round(mean_loss.item(), 3),
                   round(mean_CE_loss.item(), 3),
                   round(mean_cos_C2S1_C3S1_loss.item(), 3),
                   round(mean_cos_C1S2_C1S3_loss.item(), 3),
                   round(mean_cos_C1S2_C2S1_loss.item(), 3), )

        if not torch.isfinite(loss):
            print('\nWARNING: non-finite loss, ending training ', loss)
            non_times += 1
            if non_times == 3:
                print('Nan loss accurs more than 3 times')
                sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item(), mean_CE_loss.item(), mean_cos_C2S1_C3S1_loss.item(), mean_cos_C1S2_C2S1_loss.item(), \
           mean_cos_C1S2_C1S3_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, param_change=False, param=None, device='cuda'):
    if param_change:
        model.load_state_dict(torch.load(param), strict=False)
    model.cuda()
    model.eval()

    sum_num = torch.zeros(1).to(device)

    num_samples = len(data_loader.dataset)
    if param_change:
        print('\nValiadation dataset contains {0} items, using {1}'.format(num_samples, param))
    else:
        print('\nValiadation dataset contains {0} items'.format(num_samples))

    data_loader = tqdm(data_loader, desc="validation...")

    for test_data in data_loader:
        test_imgs, test_labels = test_data
        test_imgs = test_imgs.cuda()
        test_labels = test_labels.cuda()
        test_labels = test_labels.squeeze(dim=1)  # [B,1] -> [B]
        outputs = model(test_imgs)
        predict_y = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(predict_y, test_labels).sum().item()
    test_acc = sum_num / num_samples
    print('Testing accuracy is', round(test_acc.item(), 3))
    return test_acc.item(), sum_num.item(), num_samples


@torch.no_grad()  # if enable is_IN
def evaluate_IN(model, data_loader, param_change=False, param=None, device='cuda'):
    if param_change:
        model.load_state_dict(torch.load(param))
    model.cuda()
    model.eval()

    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    if param_change:
        print('\nValiadation dataset contains {0} items, using {1}'.format(num_samples, param))
    else:
        print('\nValiadation dataset contains {0} items'.format(num_samples))
    data_loader = tqdm(data_loader, desc="validation...")

    for test_data in data_loader:
        test_imgs, test_labels = test_data
        test_imgs = test_imgs.cuda()
        test_labels = test_labels.cuda()
        test_labels = test_labels.squeeze(dim=1)
        outputs = model(test_imgs, is_IN=False)  # ,norm_pos = None)
        predict_y = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(predict_y, test_labels).sum().item()
    test_acc = sum_num / num_samples
    print('Testing accuracy is', round(test_acc.item(), 3))
    return test_acc.item(), sum_num.item(), num_samples


def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
