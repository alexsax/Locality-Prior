# Locality-Prior

## Table of Contents 
0. Intro
0. Related Work
0. Methods
  0. Implementation
0. Experiments
  0. MNIST
  0. ImageNet
0. Results
0. Conclusions
0. References 

## Intro
---

## Related Work
---

## Methods 
---

This project proposes what we call a _locality prior_ layer. The locality prior layer is a way to impose a wiring cost between layers. The locality prior (LP) layer admits many different wiring costs and many network topologies. Specifically, the LP layer is a fully connected layer with an elementwise wiring cost between every neuron from the precious layer and every layer of the new layer. 

As a pedagogical example, we induce a 1D Euclidean geometry where the wiring cost is proportional to neuronal distance. Let's say the prior layer has 7 neurons. Then the LP layer will look as follows: 

<img width="599" alt="Figure 2" src="https://user-images.githubusercontent.com/5157485/27008388-7dfb3280-4e24-11e7-8462-5482bed9b92f.png">

Where darker colors indicate more expensive connections between neurons. As an simpler way to show these connections which define the layer, we can abbreviate the full figures with this representaiton:

<img width="255" alt="Simple LP layer" src="https://user-images.githubusercontent.com/5157485/27008395-a04af672-4e24-11e7-9548-7f1038eaf598.png">

This is a very simple example of an LP layer, but we can also create more complex topologies. The connectivity in the brain is more similar to a 2D topology than the 1D version above, since the cortical manifold is a thin sheet of gray matter which is compactly folded and wrapped around white matter in the brain. We can approximate this with a 2D topology and where distance is the standard L2 norm. We also give this a linear wiring cost. Another way of interpreting this is that the signal decays linearly with distance. The initial signal therefore needs to be stronger to compensate for the lossiness. This can be penalized with standard L1 weight deay. 

The induced topology can be visualized, just at before. Here is the transmitted signal for the center neuron:

![prior_for_center_neuron](https://user-images.githubusercontent.com/5157485/27008516-340998d4-4e28-11e7-98d0-8eb7e562d599.png)

And for the top-left neuron.

![prior_for_top_left_neuron](https://user-images.githubusercontent.com/5157485/27008518-3cb16dea-4e28-11e7-92ab-add608afff35.png)

#### Implementation
If the input layer has activations (x) of size K and the output is a vector _y_ of size K activations, then the LP layer has two matrices and a bias vector. The first one, the prior matrix _P_ is KxK and contains the transmitted signal from neuron _i_ to neuron _j_ in element _ij_. The other matrix _W_ is the standard KxK one for a FC layer and the bias _b_ is included here. The prior is multiplied with the activations after the FC layer, elementwise. The activations _A_ are computed as _A = P.*(Wx + b)_. 

The LP layer is implemented as a layer in PyTorch. The prior should be rescaled so that the total input to each neuron is the same as before, or multiplied by K/sum(inputs). The network will be able to learn from this, but this will also allow smaller weights in W which will interfere with the effectiveness of the weight regularization. Instead, we apply a Batch Normalization layer after the LP layer. In the experiments, we make sure to include the BN layer in the control networks for fair comparisons. 

## Experiments
--- 

We analyze the effects of the LP layer on two standard networks trained on two standard datasets. In particular, we train LeNet on MNIST and AlexNet on ImageNet.

| Hyperparameter | MNIST   | ImageNet |
| -------------- |:-------:|:--------:|
| Optimizer      | RMSProp | RMSProp  |
| Momentum       | 0.5     | 0.9      |
| Learning Rate  | 0.01    | 0.1      |
| Weight Decay   | 0.01    | 1e-4     |
| Epochs         | 10      | 90       |
| Batch Size     | 64      | 256      | 

### MNIST
We trained two LeNets on MNIST where the LeNets have an extra Linear and BN layer before the final FC layer. In our treatment network (with the locality prior) we change the additional Linear layer into a Locality prior layer. Our hyperparameters are given in the table. Training proceeds much the same in both networks, and activations look fairly similar as well. We tried varying the weight decay in order to make the locality prior have more impact (higher weight decays hampered performance, lower decay had no efficacy). The chosen weight decay of 0.01 was as high as we could go while still achieving high accuracy. We also tried alternative signal decay, where cost grew with the sqrt and, alternatively, the square of the distance. When the cost was quadratic with distance, performance suffered. When the cost grew with the squareroot of the distance, performance was good but there seemed to be almost no difference in activation patterns. Therfore, we chose to use a linear cost. 

| Type         | Accuracy | Loss   |
| ------------ |:--------:|:------:|
| Control      | 98.1     | 0.101  |
| Local        | 97.9     | 0.122  |

We can now visualize some of the test-set results from the two networks after training. Here are the activation patterns for each class, averaged over 1000 images from the test set. 
#### Locality input activations
<img width="827" alt="fc1_outputs" src="https://user-images.githubusercontent.com/5157485/27009093-08130d48-4e3a-11e7-8676-237af7bba256.png">

#### Locality output activations
<img width="814" alt="locality_outputs" src="https://user-images.githubusercontent.com/5157485/27009095-0e7ba6fe-4e3a-11e7-9663-f8eee2d64c9e.png">

#### Analysis
The output of the locality prior layer has activations which have a slightly but statistically reduced variance when compared to the control network. 
![variances](https://user-images.githubusercontent.com/5157485/27009241-f1f8fe4c-4e3d-11e7-9815-70387b45ee4f.png)

| Locality Variance | Control Variance |
| ----------------- | ---------------- |
| 8.458 |  8.667 |
`T-score: -6.49, P-value: 1.35e-10`


### ImageNet
We also ran the experiment on ImageNet. We changed the FC6 layer to a LP layer and, again, added a BN layer afterwards to both the treatment and control networks. Since AlexNet's FC6 layer is much larger than LeNet's, the prior is qualitatively different than in LeNet and connections are sparser. Here is what the two priors look like:

| LeNet | AlexNet |
| ----- | ------- |
| ![mnist_prior](https://user-images.githubusercontent.com/5157485/27009331-ea1344ce-4e3f-11e7-998c-3b9a940273b8.png) | ![prior_for_center_neuron](https://user-images.githubusercontent.com/5157485/27008516-340998d4-4e28-11e7-98d0-8eb7e562d599.png) |


## Results
---

Adding a wiring cost to linear layers

## References
---

[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4587756/
