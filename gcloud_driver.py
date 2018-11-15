source './google-cloud-sdk/path.bash.inc'
source './google-cloud-sdk/completion.bash.inc'
#pip install --upgrade gcloud
## Initiate Google Cloud Compute Engineer 
#gcloud compute ssh gpu-instance --zone us-east1-c

gcloud compute --project "agile-sanctum-183223" ssh --zone "us-east1-b" "instance-1"


## Install needed packages
# pip install --upgrade pip
# python -m pip install numpy scipy matplotlib pillow
# ## Pytorch on Linux
# pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
# sudo pip install torchvision
# sudo apt-get install python-tk

# mkdir HW3
# gcloud compute scp Google\ Drive/Aozhou/PhD/CS\ MSE/Computer\ Vision/HW3/lfw.tgz gpu-instance:~/HW3 --zone us-east1-c
# gcloud compute scp Google\ Drive/Aozhou/PhD/CS\ MSE/Computer\ Vision/HW3/test.txt gpu-instance:~/HW3 --zone us-east1-c
# gcloud compute scp Google\ Drive/Aozhou/PhD/CS\ MSE/Computer\ Vision/HW3/train.txt gpu-instance:~/HW3 --zone us-east1-c

gcloud compute scp test.txt gpu-instance:~/Homework3 --zone us-east1-c
gcloud compute scp train.txt gpu-instance:~/Homework3 --zone us-east1-c


cd Google\ Drive/Aozhou/PhD/CS\ MSE/Computer\ Vision/HW3
gcloud compute scp Code/RandomTransforms.py gpu-instance:HW3 --zone us-east1-c
gcloud compute scp Code/p1a.py gpu-instance:HW3 --zone us-east1-c
gcloud compute scp Code/p1b.py gpu-instance:HW3 --zone us-east1-c
#gcloud compute scp Code/p1ba.py gpu-instance:HW3 --zone us-east1-c



### Train - SiameseBCENet - No augmentation 
python p1a.py --save '1a_model_param.pt'
### Test - SiameseBCENet - No augmentation 
python p1a.py --load '1a_model_param.pt'

### Train - SiameseBCENet - with augmentation 
python p1a.py --save '1a_model_param_augmentation.pt' --transforms augment --epochN 58
### Test - SiameseBCENet - with augmentation 
python p1a.py --load '1a_model_param_augmentation.pt'



#python p1ba.py --save '1b_model_param_hlr.pt' --epochN 180

### Train - SiameseCLNet - No augmentation 
python p1b.py --save '1b_model_param.pt' --epochN 200
### Test - SiameseCLNet - No augmentation 
python p1b.py --load '1b_model_param.pt'

### Train - SiameseCLNet - with augmentation 
python p1b.py --save '1b_model_param_augmentation.pt' --transforms augment --epochN 100
### Test - SiameseCLNet - with augmentation 
python p1b.py --load '1b_model_param_augmentation.pt'


sbatch -o test.log -e test.err -p serial_requeue --wrap="python p1a.py --save '1a_model_param_test.pt' --epochN 1"


gcloud compute scp gpu-instance:~/HW3/'1a_model_param.pt' ./ --zone us-east1-c
gcloud compute scp gpu-instance:~/HW3/'1a_model_param_augmentation.pt' ./ --zone us-east1-c
gcloud compute scp gpu-instance:~/HW3/'1b_model_param.pt' ./ --zone us-east1-c
gcloud compute scp gpu-instance:~/HW3/'1b_model_param_augmentation.pt' ./ --zone us-east1-c


gcloud compute scp gpu-instance:~/HW3/'Loss_function.png' ./ --zone us-east1-c


## Free GPU memory
nvidia-smi
sudo kill -9 2454

## Exit 
exit

sudo power off # To stop running
stop


gcloud dataproc jobs list --project "agile-sanctum-183223" --zone "us-east1-c" "gpu-instance"

###### Python Testing Code ######
pad_img = Image.new("RGB", output_size)

from RandomTransforms import *
image1.show()

randomTranslation = RandomTranslation((-10,10))
randomTranslation(image1).show()

randomMirrorFlipping = RandomMirrorFlipping()
randomMirrorFlipping(image1).show()

randomRotation = RandomRotation((-30., 30.))
randomRotation(image1).show()

randomScaling = RandomScaling((0.7, 1.3))
randomScaling(image1).show()
