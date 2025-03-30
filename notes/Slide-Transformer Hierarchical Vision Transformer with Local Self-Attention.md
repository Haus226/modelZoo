# (CVPR 2023) Slide-Transformer: Hierarchical Vision Transformer with Local Self-Attention

The authors propose an improved **local attention mechanism** that is both **efficient** and **adaptive**, drawing inspiration from **convolutional operations** and **deformable networks**.


## 1. Efficient Sliding Window Attention via Depthwise Convolutions  
Traditional sliding window attention suffers from **inefficient feature shifting**. The authors replace this with **depthwise convolutions**, making the process more hardware-efficient.  

> *"We re-formulate the key and value matrix from a row-based view and show that each row corresponds to the input feature shifted in different directions."*  
*"This new insight gives
us the chance to take a further step, that allows us to replace the shifting operation with carefully designed Depthwise Convolutions."*

They reinterpret the **Im2Col operation** as a sequence of shifts:  

> *"Take k = 3 as an example,
if we first shift the original feature map towards 9 different
directions (Fig.3(2.b)), then flatten these features into rows
and finally concatenate them in column (Fig.3(2.c)), the obtained key/value matrix is proved equivalent to HW local
windows which can recover the exact same output of the
original Im2Col function (Fig.3(1.c)).*

This allows them to replace **\( k^2 \) shifting operations with depthwise convolutions**, significantly improving efficiency.

---

## 2. Reparameterization Inspired by RepVGG  
To further improve flexibility, they adopt a **parallel convolution path with learnable parameters**, similar to RepVGG’s reparameterization strategy.  

> *"At the training stage, we maintain two paths, one with designed kernel weights to perform shifting towards different directions, and the other with learnable parameters to enable more flexibility."*  
*"At the inference stage, we merge these two convolution operations into a single path with re-parameterization, which improves the model capacity while maintaining inference efficiency."*

This allows their attention mechanism to retain the **efficiency of convolutions while enhancing flexibility**.

---

## 3. Deformed Shifting Module (Inspired by Deformable Attention)  
Instead of using **fixed feature shifts**, they introduce a **learnable shifting mechanism** inspired by **Deformable Convolutions (DCN)**.  

> *"To further enhance flexibility, we introduce a novel deformed shifting module that relaxes fixed key and value positions to deformed features within the local region."*  
*"By using a re-parameterization technique, we effectively increase the model capacity while preserving inference efficiency."*

Their module dynamically **adapts sampling locations** within the local window:  

> *"The learnable convolution kernel shows a resemblance with the deformable technique in DCN. Similar to the bilinear interpolation of four neighboring pixels in DCN, our deformed shifting module can be viewed as a linear combination of features within the local window."*

This means their approach **mimics deformable attention**, making local attention more flexible in handling complex structures.

---

## Core Idea  
By **combining efficient local attention (via depthwise convolution) with adaptive spatial sampling (via learnable shifts like DCN)**, the authors improve both **efficiency** and **flexibility** in attention mechanisms, making them more effective and hardware-friendly for real-world applications. However, after reading the code provided in [repo](https://github.com/LeapLabTHU/Slide-Transformer/), I found that the first path of depthwise convolution is not freezed, therefore an issue is opened to ask for clarification.
