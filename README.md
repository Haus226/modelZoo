# modelZoo
A repository that implements proposed idea from popular conference such as ECCV, CVPR, NIPS and etc from scratch.

|Category|Number of Implementation|
|--------|------------------------|
|Activation|3|
|CNN Attention|21|
|CNN Backbone|9|
|ViT Backbone|14|
|Convolution|10|
|Normalization|1|
|Sampling/Pooling|4|

## [Variants of Convolution](/conv/README.md)
<details>
  <summary>Details</summary>

|Title|Conference/Publication|Official Repo|My Implementation|
|-----|----------------------|-------------|-----------------|
|[CondConv: Conditionally Parameterized Convolutions for Efficient Inference](http://arxiv.org/abs/1904.04971)|NIPS 2019|[Repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv)|[CondConv.py](/conv/CondConv_NIPS_2019.py)
|[Dynamic Convolution: Attention over Convolution Kernels](http://arxiv.org/abs/1912.03458)|CVPR 2020|None|[DynamicConv.py](/conv/DynamicConv_CVPR_2020.py)
|[Involution: Inverting the Inherence of Convolution for Visual Recognition](http://arxiv.org/abs/2103.06255)|CVPR 2021|[Repo](https://github.com/d-li14/involution)|[Involution.py](/conv/Involution_CVPR_2021.py)|
|[MixConv: Mixed Depthwise Convolutional Kernels](http://arxiv.org/abs/1907.09595)|BMCV 2019|[Repo]( https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet)|[MixConv.py](/conv/MixConv_BMCV_2019.py)|
|[Omni-Dimensional Dynamic Convolution](http://arxiv.org/abs/2209.07947)|ICLR 2022|[Repo](https://github.com/OSVAI/ODConv)|[ODConv.py](/conv/ODConv_ICLR_2022.py)
|[Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition](http://arxiv.org/abs/2006.11538)|Withdrawn from ICLR 2021|[Repo](https://github.com/iduta/pyconv)|[PyConv.py](/conv/PyConv.py)
|[SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy](https://ieeexplore.ieee.org/document/10204928/)|CVPR 2023|[Repo](https://github.com/cheng-haha/ScConv)|[SCConv.py](/conv/SCConv_CVPR_2023.py)|
|[Improving Convolutional Networks With Self-Calibrated Convolutions](https://ieeexplore.ieee.org/document/9156634/)|CVPR 2020|[Repo](https://github.com/MCG-NKU/SCNet)|[SelfCalibratedConv.py](/conv/SelfCalibratedConv_CVPR_2020.py)|
|[SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Weights Flipping](http://arxiv.org/abs/2003.07469)|TIP 2021|[Repo](https://github.com/JiaxiongQ/SlimConv)|[SlimConv.py](/conv/SlimConv_TIP_2021.py)|
|[WeightNet: Revisiting the Design Space of Weight Networks](https://arxiv.org/abs/2007.11823)|ECCV 2020|[Repo](https://github.com/megvii-model/WeightNet)|[WeightConv.py](/conv/WeightConv_ECCV_2020.py)|
</details>





## CNN Attention
## [CNN Backbone](/backbone_block/cnn/README.md)

<details>
  <summary>Details</summary>

|Title|Conference/Publication|Official Repo|My Implementation|
|-----|----------------------|-------------|-----------------|
|[ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions](https://arxiv.org/abs/1809.01330)|NIPS 2018|[Repo](https://github.com/GenDisc/ChannelNet)|[ChannelBlock.py](/backbone_block/cnn/ChannelBlock_NIPS_2018.py)
|[A ConvNet for the 2020s](http://arxiv.org/abs/2201.03545)|CVPR 2022|[Repo](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)|[ConvNextBlock.py](/backbone_block/cnn/ConvNextBlock_CVPR_2022.py)|
|[Diverse Branch Block: Building a Convolution as an Inception-like Unit](http://arxiv.org/abs/2103.13425)|CVPR 2021|[Repo](https://github.com/DingXiaoH/DiverseBranchBlock)|[DiverseBranchBlock.py](/backbone_block/cnn/DiverseBranchBlock_CVPR_2021.py)|
|[InceptionNeXt: When Inception Meets ConvNeXt](http://arxiv.org/abs/2303.16900)|CVPR 2024|[Repo](https://github.com/sail-sg/inceptionnext)|[InceptionNextBlock.py](/backbone_block/cnn/InceptionNextBlock_CVPR_2024.py)|
|[Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)|CVPR 2015|Most of the Framework including this model|[InceptionV1Block.py](/backbone_block/cnn/InceptionV1Block_CVPR_2015.py)|
|[RepVGG: Making VGG-style ConvNets Great Again](http://arxiv.org/abs/2101.03697)|CVPR 2021|[Repo](https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)|[RepVGGBlock.py](/backbone_block/cnn/RepVGGBlock_CVPR_2021.py)|
|[Data-Driven Neuron Allocation for Scale Aggregation Networks](http://arxiv.org/abs/1904.09460)|CVPR 2019|[Repo](https://github.com/Eli-YiLi/ScaleNet/tree/master)|[ScaleBlock.py](/backbone_block/cnn/ScaleBlock_CVPR_2019.py)|
|[Visual Attention Network](http://arxiv.org/abs/2202.09741)|CVM 2023|[Repo](https://github.com/Visual-Attention-Network/VAN-Classification)|[VANBlock.py](/backbone_block/cnn/VANBlock_CVM_2023.py)|
|[VanillaNet: the Power of Minimalism in Deep Learning](http://arxiv.org/abs/2305.12972)|None|[Repo](https://github.com/huawei-noah/VanillaNet/blob/main/models/vanillanet.py)|[VanillaBlock.py](/backbone_block/cnn/VanillaBlock.py)|


</details>

## [ViT Backbone](/backbone_block/vit/README.md)

<details>
  <summary>Details</summary>

</details>

## [Normalization](/normalization/README.md)
<details>
  <summary>Details</summary>

|Title|Conference/Publication|Official Repo|My Implementation|
|-----|----------------------|-------------|-----------------|
|[Bnet: Batch normalization with enhanced linear transformation](http://arxiv.org/abs/2011.14150)|TPAMI 2023|[Repo](https://github.com/yuhuixu1993/BNET)|[BNet.py](/normalization/BNet%20(TPAMI%202023).py)|
</details>

## Activation
## [Sampling](/sampling/README.md)
<details>
  <summary>Details</summary>

|Title|Conference/Publication|Official Repo|My Implementation|
|-----|----------------------|-------------|-----------------|
|[Making Convolutional Networks Shift-Invariant Again](http://arxiv.org/abs/1904.11486)|ICML 2019|[Repo](https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py)|[BlurPool.py](/sampling/BlurPool%20(ICML%202019).py)|
|[CARAFE: Content-Aware ReAssembly of FEatures](http://arxiv.org/abs/1905.02188)|ICCV 2019|[Unofficial Repo](https://github.com/XiaLiPKU/CARAFE/blob/master/carafe.py)|[CARAFE.py](/sampling/CARAFE%20(ICCV%202019).py)(Pasted from Repo)|
|[LIP: Local Importance-based Pooling](http://arxiv.org/abs/1908.04156)|ICCV 2019|[Repo](https://github.com/sebgao/LIP)|[LIP.py](/sampling/LIP%20(ICCV%202019).py)|
|[Refining activation downsampling with SoftPool](http://arxiv.org/abs/2101.00440)|ICCV 2021|[Repo](https://github.com/alexandrosstergiou/SoftPool)|[SoftPool.py](/sampling/SoftPool%20(ICCV%202021).py)
</details>



