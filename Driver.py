source './google-cloud-sdk/path.bash.inc'
source './google-cloud-sdk/completion.bash.inc'
#pip install --upgrade gcloud
## Initiate Google Cloud Compute Engineer 
#gcloud compute ssh ad-prediction --zone us-east1-c
gcloud compute --project "agile-sanctum-183223" ssh --zone "us-east1-b" "instance-1"

## Install needed packages
# pip install --upgrade pip
# sudo python -m pip install numpy scipy matplotlib pillow
# ## Pytorch on Linux
# sudo pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
# sudo pip install torchvision
# sudo apt-get install python-tk

# curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
#   sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
#   sudo apt-get update
#   sudo apt-get install -y cuda python-pip python-dev build-essential
#   sudo pip install --upgrade pip virtualenv
#   pip install --user http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
#   pip install --user torchvision scipy opencv-python matplotlib jupyter scikit-image nbstripout nibabel

#gcloud compute scp --recurse /Users/waz/JHU/CV-ADNI/ImageNoSkull ad-prediction:~/ --zone us-east1-c
gcloud compute scp Code/AD_Prediction/test_2classes.txt instance-1:~/ --zone us-east1-b
gcloud compute scp Code/AD_Prediction/train_2classes.txt instance-1:~/ --zone us-east1-b

### Upload the data to google cloud
gcloud compute scp --recurse /Users/waz/Google\ Drive/Deep\ learning\ project/Whole instance-1:~/ --zone us-east1-b


# gcloud compute scp Code/AD_Prediction/AlexNet2D.py ad-prediction: --zone us-east1-b
# gcloud compute scp Code/AD_Prediction/AlexNet3D.py ad-prediction: --zone us-east1-b
# gcloud compute scp Code/AD_Prediction/custom_transform.py ad-prediction: --zone us-east1-b
# gcloud compute scp Code/AD_Prediction/AD_2DSlicesData.py ad-prediction: --zone us-east1-b
# gcloud compute scp Code/AD_Prediction/AD_Dataset.py ad-prediction: --zone us-east1-b
# gcloud compute scp Code/AD_Prediction/main_alextnet.py ad-prediction: --zone us-east1-b



# gcloud compute scp gcloud/custom_transform2D.py ad-prediction: --zone us-east1-b
# gcloud compute scp gcloud/AD_2DSlicesData.py ad-prediction: --zone us-east1-b
# gcloud compute scp gcloud/main_alextnet.py ad-prediction: --zone us-east1-b
# gcloud compute scp gcloud/AlexNet2D.py ad-prediction: --zone us-east1-b


# gcloud compute scp gcloud/main_cnn_autoencoder.py ad-prediction: --zone us-east1-c
# gcloud compute scp gcloud/autoencoder.py ad-prediction: --zone us-east1-c
# gcloud compute scp gcloud/AD_3DRandomPatch.py ad-prediction: --zone us-east1-c


# gcloud compute scp gcloud/autoencoder_pretrained_model39 ad-prediction: --zone us-east1-c

# gcloud compute scp gcloud/AD_Standard_CNN_Dataset.py ad-prediction: --zone us-east1-c
# gcloud compute scp gcloud/autoencoder.py ad-prediction: --zone us-east1-c
# gcloud compute scp gcloud/cnn_3d_with_ae.py ad-prediction: --zone us-east1-c
# gcloud compute scp gcloud/main_cnn_autoencoder.py ad-prediction: --zone us-east1-c

gcloud compute scp Code/custom_transform.py instance-1: --zone us-east1-b
gcloud compute scp Code/AD_Dataset.py instance-1: --zone us-east1-b
gcloud compute scp Code/AD_Standard_2DRandomSlicesData.py instance-1: --zone us-east1-b
gcloud compute scp Code/AD_Standard_2DTestingSlices.py instance-1: --zone us-east1-b
gcloud compute scp Code/AD_Standard_2DSlicesData.py instance-1: --zone us-east1-b
gcloud compute scp Code/AlexNet2D.py instance-1: --zone us-east1-b
gcloud compute scp Code/custom_transform2D.py instance-1: --zone us-east1-b
gcloud compute scp Code/main_alexnet.py instance-1: --zone us-east1-b
#gcloud compute scp gcloud/rename.py ad-prediction: --zone us-east1-c

gcloud compute scp train_2classes.txt instance-1: --zone us-east1-b
gcloud compute scp test_2classes.txt instance-1: --zone us-east1-b


#gcloud compute scp Code/p1ba.py ad-prediction:HW3 --zone us-east1-c
python main_alexnet.py --network_type AlexNet2D --optimizer Adam --learning_rate 1e-3 --save stand_ALEX_Last_cov_non_random_best_model --batch_size 16 --gpuid 0 --epochs 100



gcloud compute scp ad-prediction:~/'Loss_function1.png' ./ --zone us-east1-c


## Free GPU memory
nvidia-smi
sudo kill -9 2454

## Exit 
exit

sudo power off # To stop running
stop


gcloud dataproc jobs list --project "agile-sanctum-183223" --zone "us-east1-c" "ad-prediction"


