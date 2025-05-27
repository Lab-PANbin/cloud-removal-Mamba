# cloud-removal-Mamba

Installation

conda create -n crfamba python=3.10.14

conda activate crfamba

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm

pip install causal-conv1d

pip install mamba-ssm==1.2.0

cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

We have uploaded the experimental results from the paper.

 (通过网盘分享的文件：tcloud_CR-Famba.zip 链接: https://pan.baidu.com/s/1AMsHf2WSkV0WuyKZA9APjA 提取码: yyrz 
--来自百度网盘超级会员v5的分享 )

To reproduce PSNR/SSIM scores of the paper, run

evaluate_PSNR_SSIM.m (https://github.com/liujiaocv/cloud-removal/blob/main/CMNet/evaluate_PSNR_SSIM.m)

If you use CR-Famba, please consider citing:

J. Liu, B. Pan and Z. Shi, "CR-Famba: A Frequency-Domain Assisted Mamba for Thin Cloud Removal in Optical Remote Sensing Imagery," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2025.3542976.
