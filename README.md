How to Reproduce the Experiments
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

After downloading the images, the CAMVID_ROOTconsists of the following.

> CAMVID_ROOT

    class_dict.csv    
    test    
    test_labels    
    train    
    train_labels    
    val    
    val_labels
    
[Camvid official site]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[kaggle site]: https://www.kaggle.com/carlolepelaars/camvid

# 2. Python Environment Configuration
It is recommended to use Anaconda to create an execution environment.

## Anaconda Virtual Environment (optional)
The code was implemented and tested in Python 3.7. It is not necessary because it can be operated on python 3 or higher.

Run the following command to create a virtual environment in Anaconda, please.

    conda create --name repr python=3.7

Please activate the virtual environment after the installation is complete.

    conda activate repr

## Python Package Installation
it is necessary for two special packages to install manually. Except that, required modules(tqdm, tensorboardX) are listed in requirements.txt and can be installed with the following command.

    pip install -r requirements.txt

### PyTorch
The phytorch framework was used to implement the deep learning network.
PyTorch needs to be installed according to the CUDA environment from the [official site].

The experiment was conducted using PyTorch 1.6.0 in CUDA 10.2 and CUDNN 7.6.5.
If the same CUDA version is used, it can configure the experiment environment using the command below.

    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

[official site]: https://pytorch.org/

### Apex
The apex package was used to expand NVIDA DGX.
Please install apex from https://www.github.com/nvidia/apex to run this program.

The apex needs to be downloaded and installed by executing the following command.

    
    git clone https://github.com/NVIDIA/apex.git
    cd apex
    python setup.py install

# 3. Training and Validation
To proceed with the experiment, the following 4 arguments need to be set.

    train_data: Training dataset path. It should be CITYSCAPES_ROOT or CAMVID_ROOT
    val_data: Validation dataset path. It should be CITYSCAPES_ROOT or CAMVID_ROOT
    l4_module: A module applied to the base network output(output stride 16). It should be selected from None, DeformConv2d, SqzDeformConv, SPP, ASPP and DSPP.
    l3_module: A module applied to the output stride 8. It should be selected from None, SPP, ASPP and DSPP.)

To set up as shown in Table 3, 4 and 5, give the following values please.

    Original                              : --l4_module None --l3_module None
    Deformable Convolution                : --l4_module 'DeformConv2d' --l3_module None
    Squeezed Deformable Convolution (ours): --l4_module 'SqzDeformConv' --l3_module None
    SPP                                   : --l4_module 'SPP' --l3_module None
    ASPP                                  : --l4_module 'ASPP' --l3_module None
    DSPP (ours)                           : --l4_module 'DSPP' --l3_module None
    SPP Extension                         : --l4_module 'SPP' --l3_module 'SPP'
    ASPP Extension                        : --l4_module 'ASPP' --l3_module 'ASPP'
    Extended DSPP (ours)                  : --l4_module 'DSPP' --l3_module 'DSPP'

## Single GPU




