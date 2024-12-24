# OCD(On-The-Fly Category Discovery)
CVPR 2023.Ruoyi Du, Dongliang Chang, Kongming Liang*, Timothy Hospedales, Yi-Zhe Song, Zhanyu Ma
# Introduction
一种动态类别发现方法，旨在使模型即时意识到新的类别样本（能够进行归纳学习和流式推理）。该方法首先设计了一个基于散列编码的可扩展模型作为实际的基线即main_baseline。随后，注意到散列码对类内方差的敏感性，进一步提出了一种新的符号-幅度解纠缠结构即main_SMILE，来减轻它（散列码对类内方差的敏感性）带来的干扰。本文使用的基础模型为ViT-B-16。
A dynamic category discovery method has been devised with the primary objective of enabling the model to promptly assimilate new category samples, demonstrating capabilities in inductive learning and streaming reasoning. Initially, a scalable model, referred to as main_baseline, has been formulated based on hash coding as a practical foundation. Recognizing the susceptibility of hash codes to intra-class variability, a novel sign-amplitude disentanglement structure, denoted as main_SMILE, has been introduced to alleviate the disruptions arising from this source of interference. The foundational model employed in this study is the ViT-B-16 architecture.
# Eviroment Requirement
python                    3.8.17  
torch                     2.0.0+cu118 
tqdm                      4.65.0                 
transformers              4.32.0  
# Dataset
Three coarse-grained classification datasets: CIFAR10, CIFAR100, Imagenet-100, and three fine-grained classification
datasets: CUB-200-2011, Stanford Cars, and Herbarium19.
# Code explanation
 Data directory: contain dataset split and data transform/
 Logs directory: Experiment Log
 pretrained_models directory: contain predefined base model----ViT-B-16
 save directory: save the experiment result
 utils directory:
 
# main_baseline
在这一训练过程中，

