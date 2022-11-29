from .RS_getdata_new import RS_dataset
import torch
from torch.utils.data import DataLoader as Dataloader
import matplotlib.pyplot as plt

def pred_fig(net,num_plot,
            images_dir=None,
            transform=None,
            device=None):

    #index to label (0-->baseball_field)
    label_path = '/home/yrk/remote sensing dataset/DOTA_class/label_dict.txt'
    label_info = []
    with open(label_path, "r") as rd:
        for line in rd.readlines():
            line = line.strip()
            if len(line) > 0:
                split_info = [i for i in line.split(" ") if len(i) > 0]
                pred_idx, class_name = split_info
                label_info.append([pred_idx, class_name])

    dataset_dir = '/home/yrk/remote sensing dataset/DOTA_class/DOTA_test/'
    datafile = RS_dataset(dataset_dir)
    datafile_loader = Dataloader(datafile, batch_size=12, num_workers=8, shuffle=True)


    model = net
    for step,(test_imgs,test_labels) in enumerate(datafile_loader):
        test_imgs = test_imgs.cuda()
        # print(test_imgs.size())
        test_labels = test_labels.cuda()
        test_labels = test_labels.squeeze(dim=1)

        outputs = model(test_imgs)
        probs, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
        num_imgs = num_plot
        fig = plt.figure(figsize=(num_imgs * 2.5, 3), dpi=100)
        for i in range(num_imgs):
            ax = fig.add_subplot(1, num_imgs, i + 1, xticks=[], yticks=[])

            # CHW -> HWC
            npimg = test_imgs[i].cpu().numpy().transpose(1, 2, 0)
            test_label = test_labels[i].cpu().numpy()
            test_label = int(test_label)
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            plt.imshow(npimg.astype('uint8'))

            title = "{}, {:.2f}%\n(label:{})".format(
                label_info[int(preds[i])][1],  # predict class
                probs[i] * 100,  # predict probability
                label_info[test_label][1]  # true class
            )
            ax.set_title(title, color=("green" if preds[i] == test_label else "red"))
        if step == 0:
            break

    return fig
if __name__  == '__main__':
    pass