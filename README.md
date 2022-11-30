# TSCM Triplet Style-Content Metric

This repository includes introductions and implementation of ***Style-Content Metric Learning for Multidomain Remote Sensing Object Recognition*** in PyTorch.


# Datasets

We conduct experiments using four remote sensing datasets: ***[NWPU VHR-10 (Cheng, Zhou, and Han 2016)](https://gcheng-nwpu.github.io/#Datasets), [DOTA (Xia et al. 2018)](https://captain-whu.github.io/DOTA/), 
[HRRSD (Zhang et al. 2019)](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset) and [DIOR (Li et al. 2020c)](http://www.escience.cn/people/JunweiHan/DIOR.html)***.

Remote sensing objects are cut out from object detection ground truth.

Ten common object categories among four datasets are reserved for experiments, i.e., ***Airship, Ship, Storage Tank, Baseball Diamond, Tennis Court, Basketball Court, Ground Track Field, Harbor, Bridge and 
Vehicle*** .

Specifically, folder index and categories are as follows:

>01 baseball_field  
02 basketball_field  
03 overpass  
04 stadium  
05 harbor  
08 plane  
10 ship  
11 car  
13 oil_tank  
15 tennis_court  

You can download post-processed datasets from these links( google drive ):   
**[NWPU-RSOR](https://drive.google.com/file/d/1avSDZeNtys5vGNgN8JqwA-4l2CI7Oc9v/view?usp=share_link),**  
**[DOTA-RSOR](https://drive.google.com/file/d/1EzV3mgxYwDhMuAD1SwNi7JGOvtyGZoBl/view?usp=share_link),**  
**[HRRSD-RSOR](https://drive.google.com/file/d/1ZlInqLBa0nugPvIdr8DUjUcgIYHaDcEP/view?usp=share_link),**  
**[DIOR-RSOR](https://drive.google.com/file/d/1xu_hhQpdUVkwY02oQwrMrtVlDPZSiDvV/view?usp=share_link).**  

## File Structure
Trained model are saved in **Ready Model** folder, unzip it from .zip to .pth

>TSCM  
├── data  
│   ├── DIOR_RSOR  
│   │   ├── test  
│   │   ├── train  
│   │   ├── eval  
│   │   └── label_dict.txt  
│   ├── DOTA_RSOR  
│   ├── HRRSD_RSOR  
│   └── NWPU_RSOR  
├── label_dic_10.npy  
├── Modules  
├── Ready Model  
├── save  
├── train_and_test.py  
├── train.py  
├── eval.py  
└── version_check.py  


# Requirements

>PyTorch >= 1.3.1  
TorchVision >= 0.4.2  
cv2 >= 3.4.2  
>>Recommended  
tqdm >= 4.61  
[apex >= 0.1](https://github.com/NVIDIA/apex)  


# Train and Eval

## Train
For detailed argparse params, run
> python train.py -h

If you don't intend to customize params and paths, run  
> python train.py

## Eval
For detailed argparse params, run
> python eval.py -h 

If you don't intend to customize params and paths, run  
> python train.py
