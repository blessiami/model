How to reproduce the experiments
=============
# 1. Dataset Preparation
## Cityscapes
It needs to log in to download the [Cityscapes datasets][cityscapes_login].
To train and validate networks, download two zip files please.

  gtFine_trainvaltest.zip
  
  leftImg8bit_trainvaltest.zip

Let's call the cityscapes dataset path CITYSCAPES_ROOT.
If you unzip both compressed files under CITYSCAPES_ROOT, the folder has two subfolder like this.

> CITYSCAPES_ROOT  

    gtFine_trainvaltest

    leftImg8bit_trainvaltest
    
[cityscapes_login]: https://www.cityscapes-dataset.com/login/ "Go to the Cityscapes download site"

## Camvid
Now(Nov 24, 2020), on the [Camvid official site], the extracted image link is disconnected.
It needs to download and sample the video to construct datasets.
Therefore, it is recommended to use the dataset from the [kaggle site] with the extracted images.

[Camvid official site]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[kaggle site]: https://www.kaggle.com/carlolepelaars/camvid

After downloading the images, the CAMVID_ROOTconsists of the following.

> CAMVID_ROOT

    class_dict.csv
    
    test
    
    test_labels
    
    train
    
    train_labels
    
    val
    
    val_labels






