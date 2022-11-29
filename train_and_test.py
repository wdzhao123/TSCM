from Modules.RS_getdata_ten_class import RS_dataset_ten, RS_dataset_single_class_ten, RS_dataset_trinity_ten, \
    reid_folder
from torch.utils.data import DataLoader as DataLoader
from Modules.resnet_init_net import *
import time
import numpy as np
import os
from Modules.train_eval_utils import *
from apex import amp


def train(alpha=0.0, beta=0.0, gamma=0.0, save_path='',
          train_path='', test_path='',
          epochs=0,
          initial_resnet_path=''
          ):
    file = open(save_path + '/acc.txt', 'w')
    file.close()

    t1 = time.time()
    print("Begin preparing dataset.\n")
    train_data = RS_dataset_trinity_ten(dir=train_path,
                                        image_size=256)
    train_loader = DataLoader(train_data, batch_size=36,
                              shuffle=True, num_workers=8)
    print(f'It takes {time.time() - t1} seconds to prepare dataset.\n')

    test_data = RS_dataset_ten(dir=test_path, image_size=256)
    test_loader = DataLoader(test_data, batch_size=128,
                             shuffle=True, num_workers=8)

    model = resnet50_tscm(is_IN=True)
    model.load_state_dict(torch.load(initial_resnet_path)
                          , strict=False)
    model.cuda()

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.000125, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    model, optimizer = amp.initialize(models=model, optimizers=optim, opt_level='O1')

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        loss = train_one_epoch_IN(model, optim, train_loader, epoch, alpha=alpha, beta=beta, gamma=gamma)
        lr_scheduler.step(epoch)

        acc, num, all_num = evaluate_IN(model, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        torch.save(model.state_dict(), '{0}/{1}_{2}.pth'.format(save_path, epoch, int(acc * 100)))

        with open(save_path + '/acc.txt', 'a') as f:
            print('~' * 50, file=f)
            print(f'\nepoch {epoch}, acc {acc}', file=f)
            f.close()

        torch.save(model.state_dict(), f'{save_path}/{epoch}_{int(acc * 100)}.pth')

    file = open(save_path + '/acc.txt', 'a')
    file.write('\nBest acc is {}, best epoch is {}'.format(best_acc, best_epoch))
    file.close()


def test_single_class(test_path, save_path, model_path):
    label_dic = np.load('label_dic_10.npy',
                        allow_pickle=True).item()  # {'01': 'baseball_field',...
    model = resnet50_tscm(is_IN=False)

    model.load_state_dict(torch.load(model_path))

    output_dict_IN = {}

    total_acc_num_IN = 0

    total_img_num = 0

    for dir1_15 in os.listdir(test_path):
        dir1_15_dataset = RS_dataset_single_class_ten(test_path + '/' + dir1_15, image_size=256)
        dir1_15_dataloader = DataLoader(dir1_15_dataset, batch_size=128, shuffle=True, num_workers=8)

        acc, num_acc, num_all = evaluate(model=model, data_loader=dir1_15_dataloader)

        total_acc_num_IN += num_acc
        total_img_num += num_all
        output_now_IN = ['Class:{}'.format(label_dic[reid_folder(dir1_15)]), 'Accuracy:{}'.format(acc),
                         "Accurate num:{}".format(num_acc), 'This class all img number:{}'.format(num_all)]
        output_dict_IN['{}'.format(dir1_15)] = output_now_IN

    total_acc_IN = total_acc_num_IN / total_img_num

    b = [str(i) for i in bubbleSort(arr=[int(i) for i in output_dict_IN])]  # 把output_dic_init中的key按照从小到大顺序排序
    idx = ['0' + i if int(i) < 10 else i for i in b]
    print(output_dict_IN)

    if not os.path.exists(save_path + '/acc_classes.txt'):
        file = open(save_path + '/acc_classes.txt', 'w')
        file.close()

    file = open(save_path + '/acc_classes.txt', 'a')
    file.write('\n')
    file.write('~' * 50)
    file.write('\nTest data from {}.\n'.format(test_path))
    file.write('\nTotal acc：{}, Total acc num：{}, Total img num：{}\n'.format(total_acc_IN, total_acc_num_IN, total_img_num))
    file.close()
    for i in idx:
        file = open(save_path + '/acc_classes.txt', 'a')
        file.write('\n')
        file.write('{},'.format(output_dict_IN[i]))
        file.close()


if __name__ == '__main__':
    save_path = './save'


    train_path = 'DIOR_train'
    test_path = 'DIOR_test'
    initial_resnet_path = './Ready Model/resnet50-pretrained.pth'
    train(alpha=0.1, beta=0.5, gamma=0.5,
          save_path='',
          train_path='', test_path='',
          epochs=0,
          initial_resnet_path=''
          )


    nwpu_path = 'NWPU_test'
    hrrsd_path = 'HRRSD_test'
    dota_path = 'DOTA_test'
    model_path = './Ready Model/ResNet-50-TSCM.pth'
    for data in [nwpu_path, hrrsd_path, dota_path]:
        test_single_class(test_path=data,save_path=save_path,model_path=model_path)

    pass
