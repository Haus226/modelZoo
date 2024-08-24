Concept

- Compute channel attention and spatial attention at two separate branches (in parallel)
    
    - Channel Attention
        
        - Pass the feature map through AvgPool followed by a MLP with one hidden layer. The hidden activation size is $C/r$ where $C$ is the input feature of the first linear layer of MLP network and the hyperparameter $r$,  reduction ratio of neurons to controls the capacity and overhead.
        - Batch normalization is applied to adjust the scale with the spatial branch output.
        - $M_c(F)=BN(MLP(AvgPool(F)))$
        - $MLP=W_1(W_0(\cdot)+b_0)+b_1$
        - where $W_0\in\mathbb{R}^{C/r\times C},b_0\in\mathbb{R}^{C/r},W_1\in\mathbb{R}^{C\times C/r},b_1\in\mathbb{R}^{C}$
    - Spatial Attention
        
        - Reduce the channel from $C$ to $C/r$ using $1\times1$ convolution, the reduction ratio $r$ is same as the channel branch for simplicity.
        - After that, two $3\times3$ dilated convolutions are used. The dilation value $d$ is another hyperparameter.
        - Finally the channels are reduced to a spatial attention map that $\in\mathbb{R}^{1\times H\times W}$ using $1\times1$ convolution followed by a batch normalization for scale adjustment.
        - $M_s(F) = \text{BN}\left(f^{1\times1}_3\left(f^{3\times3}_2\left(f^{3\times3}_1\left(f^{1\times1}_0(F)\right)\right)\right)\right)$
    - Combination
        
        - The paper empirically verify that element-wise summation results in the best performance compared to element-wise multiplication and max operation.
        - $F’=F+F\otimes\sigma(M_c(F)+M_s(F))$
        - where $\sigma$ is sigmoid function

The official code differs significantly from the architecture described in the paper.