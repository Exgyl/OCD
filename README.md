# OCD(On-The-Fly Category Discovery)
CVPR 2023.Ruoyi Du, Dongliang Chang, Kongming Liang*, Timothy Hospedales, YiZhe Song, Zhanyu Ma
# Introduction
A dynamic category discovery method has been devised with the primary objective of enabling the model to promptly assimilate new category samples, demonstrating capabilities in inductive learning and streaming reasoning. Initially, a scalable model, referred to as main_baseline, has been formulated based on hash coding as a practical foundation. Recognizing the susceptibility of hash codes to intra-class variability, a novel sign-magnitude disentanglement structure, denoted as main_SMILE, has been introduced to alleviate the disruptions arising from this source of interference. The foundational model employed in this study is the ViT-B-16 architecture.
# Eviroment Requirement
python                    3.8.17  <br>
torch                     2.0.0+cu118 <br>
tqdm                      4.65.0 <br>                
transformers              4.32.0
# Dataset
Three coarse-grained classification datasets: CIFAR10, CIFAR100, Imagenet-100, and three fine-grained classification
datasets: CUB-200-2011, Stanford Cars, and Herbarium19.
# Code explanation
 Data directory: contain dataset split and data transform <br>
 Logs directory: Experiment Log <br>
 pretrained_models directory: contain predefined base model----ViT-B-16 <br>
 save directory: save the experiment result <br>
 utils directory: contain get dataset, parser, Indicator calculation <br>
 config.py: data directory <br>
 Contrastive.py: contrastive learning for inductive learning <br>
 vision_transformer: base model vit <br>
 main_baseline.py: baseline <br>
 main_SMILE: a novel sign-magnitude disentaglement structure
# The difference between baseline and SMILE
Features embedding is different. In the Smile framework, an additional computation involves deriving the embeddings for the sign and magnitude components. The feature embedding within the Smile framework is then determined by taking the dot product of these two embeddings. 

# Model Results
all metrics is under ACC but two evaluation protocols are Greedy-Hungarian and Strict-Hungarian
train data are split to two parts: old and new





