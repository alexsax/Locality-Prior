# Locality Prior
## A wiring cost for neural networks

## Table of Contents 
```
1. Intro
2. Methods
    2.1. Implementation
3. Experiments
    3.1. MNIST
    3.2. ImageNet
4. Results
    4.1. MNIST
    4.2. ImageNet
5. Conclusions
6. References 
```

## 1. Intro

FMRI images of the brain reveal that there is significant neuronal clustering of activations. It is believed that these correlated activations indicate areas of the brain with functional specificity. These areas function at various levels of abstraction, from low-level sensorimotor tasks, to mid-level identification of faces, and even very abstract things such as thinking about another's thoughts [2]. The The brain has specific activation patterns in response to different visual inputs, and the object being viewed can be decoded from these patterns [3]. It still an open debate as to what extent these representations are distributed versus local. 

It is still not completely known why we see functional specificity in the brain. If an engineer were to design the brain, she might create different modules which each have a different function, and connect them up in order to achieve a larger purpose. This is one way in which functional specificity could arise. 

Another reason for functional specificity is that it could be energetically favorable. It is related to Hebb's postulate [5], summarized by Löwel, that  _"Cells that fire together, wire together"_. Succintly, if two neurons need to transmit information, it requires less energy to transmit the signal if the neurons are close together. This also reduces latency. If these two neurons are connected, then, it would be beneficial for them to be in close physical proximity. 

One way to test whether such a wiring cost could lead to functional specificity is to build computational models of the brain, which are simple enough that we can simulate them quickly and exactly, yet complex enough to capture the patterns that we see in a real brain. DiCarlo et al. [6] propose that this will be the major pathway for understanding the brain in the near-term. Yamins et al. [7] find that Convolutional Neural Networks (CNNs) well estimate patterns seen in the brain, and also achieve similar performance on some visual tasks.

In this spirit, we propose a way to extend a CNN's linear layers in order to to include the wiring cost. We show that these extended models have activations which display more class-dependent neuronal clustering when compared to the standard (control) CNN models. 

---

## 2. Methods 

This project proposes what we call a _locality prior_ layer. The locality prior layer is a way to impose a wiring cost between layers. The locality prior (LP) layer admits many different wiring costs and many network topologies. Specifically, the LP layer is a fully connected layer with an elementwise wiring cost between every neuron from the precious layer and every layer of the new layer. 

As a pedagogical example, we induce a 1D Euclidean geometry where the wiring cost is proportional to neuronal distance. Let's say the prior layer has 7 neurons. Then the LP layer will look as follows: 

<img width="599" alt="Figure 2" src="https://user-images.githubusercontent.com/5157485/27008388-7dfb3280-4e24-11e7-8462-5482bed9b92f.png">

Where darker colors indicate more expensive connections between neurons. As an simpler way to show these connections which define the layer, we can abbreviate the full figures with this representaiton:

<img width="255" alt="Simple LP layer" src="https://user-images.githubusercontent.com/5157485/27008395-a04af672-4e24-11e7-9548-7f1038eaf598.png">

This is a very simple example of an LP layer, but more complex topologies are also possible. The connectivity in the brain is more similar to a 2D topology than the 1D version above, since the cortical manifold is a thin sheet of gray matter which is compactly folded and wrapped around white matter in the brain. We can approximate this with a 2D topology and where distance is the standard L2 norm. We also give this a linear wiring cost. Another way of interpreting this is that the signal decays linearly with distance. The initial signal therefore needs to be stronger to compensate for the lossiness. This can be penalized with standard L1 weight deay. 

The induced topology can be visualized, just at before. Here is the transmitted signal for the center and top-left neuron:

| Prior foor center neuron | Prior for top-left neuron |
| ------------------------ | ------------------------- |
| ![prior_for_center_neuron](https://user-images.githubusercontent.com/5157485/27009386-689ec632-4e41-11e7-9c9d-8fb0afe38f63.png) | ![prior_for_top_left_neuron](https://user-images.githubusercontent.com/5157485/27008518-3cb16dea-4e28-11e7-92ab-add608afff35.png) |
| The above image shows how the wiring cost increases linearly with Euclidean distance in our topology | The topology is a 2D grid with no wraparound, and is threrefore a closed cutout of the plane -- not homeomorphic to a sphere |


### 2.1. Implementation
If the input layer has activations (x) of size K and the output is a vector _y_ of size K activations, then the LP layer has two matrices and a bias vector. The first one, the prior matrix _P_ is KxK and contains the transmitted signal from neuron _i_ to neuron _j_ in element _ij_. The other matrix _W_ is the standard KxK one for a FC layer and the bias _b_ is included here. The prior is multiplied with the activations after the FC layer, elementwise. The activations _A_ are computed as _A = P.*(Wx + b)_. 

The LP layer is implemented as a layer in PyTorch. The prior should be rescaled so that the total input to each neuron is the same as before, or multiplied by K/sum(inputs). The network will be able to learn from this, but this will also allow smaller weights in W which will interfere with the effectiveness of the weight regularization. Instead, we apply a Batch Normalization layer after the LP layer. In the experiments, we make sure to include the BN layer in the control networks for fair comparisons. 

Note: This could probably also be implemented as a convolutional layer, too. 

---

## 3. Experiments

We analyze the effects of the LP layer on two standard networks trained on two standard datasets. In particular, we train LeNet on MNIST and AlexNet on ImageNet. We calculate the variance of the activations in both the inputs and outputs of the LP layer. We show that in 3 of the 4 cases, the activations have significantly lower variance. In the case where variance is higher, this is caused by one particular class with a several highly indicative neurons, spread out. 

| Hyperparameter | MNIST   | ImageNet |
| -------------- |:-------:|:--------:|
| Optimizer      | RMSProp | RMSProp  |
| Momentum       | 0.5     | 0.9      |
| Learning Rate  | 0.01    | 0.1      |
| Weight Decay   | 0.01    | 1e-4     |
| Epochs         | 10      | 90       |
| Batch Size     | 64      | 256      | 

### 3.1. MNIST
We trained two LeNets on MNIST where the LeNets have an extra Linear and BN layer before the final FC layer. In our treatment network (with the locality prior) we change the additional Linear layer into a Locality prior layer. Our hyperparameters are given in the table. Training proceeds much the same in both networks, and activations look fairly similar as well. We tried varying the weight decay in order to make the locality prior have more impact (higher weight decays hampered performance, lower decay had no efficacy). The chosen weight decay of 0.01 was as high as we could go while still achieving high accuracy. We also tried alternative signal decay, where cost grew with the sqrt and, alternatively, the square of the distance. When the cost was quadratic with distance, performance suffered. When the cost grew with the squareroot of the distance, performance was good but there seemed to be almost no difference in activation patterns. Therfore, we chose to use a linear cost. 

| Type         | Accuracy | Loss   |
| ------------ |:--------:|:------:|
| Control      | 98.1     | 0.101  |
| Local        | 97.9     | 0.122  |


### 3.2. ImageNet
We also ran the experiment on ImageNet. We changed the FC6 layer to a LP layer and, again, added a BN layer afterwards to both the treatment and control networks. Since AlexNet's FC6 layer is much larger than LeNet's, the prior is qualitatively different than in LeNet and connections are sparser. Here is what the two priors look like:

| LeNet | AlexNet |
| ----- | ------- |
| ![mnist_prior](https://user-images.githubusercontent.com/5157485/27009331-ea1344ce-4e3f-11e7-998c-3b9a940273b8.png) | ![prior_for_center_neuron](https://user-images.githubusercontent.com/5157485/27009386-689ec632-4e41-11e7-9c9d-8fb0afe38f63.png) |


--- 

## 4. Results

### 4.1. MNIST
We can now visualize some of the test-set results from the two networks after training. Here are the activation patterns for each class, averaged over 1000 images from the test set. We find that the output activations of the LP layer display more clustering, bcompared to the control network, but any difference is hard to see visually. 

#### Locality input activations
<img width="827" alt="fc1_outputs" src="https://user-images.githubusercontent.com/5157485/27009093-08130d48-4e3a-11e7-8676-237af7bba256.png">
Here, digit 3 is the one that causes the local variance to be higher. When digit 3 is dropped, the difference is no longer significant. In addition, note that the prior applied to LeNet does not enforce locality as well as it does for AlexNet, since LeNet has a very small linear layer.

#### Locality output activations
<img width="814" alt="locality_outputs" src="https://user-images.githubusercontent.com/5157485/27009095-0e7ba6fe-4e3a-11e7-9663-f8eee2d64c9e.png">

The output of the locality prior layer has activations which have a slightly but statistically reduced variance when compared to the control network. 

![variances](https://user-images.githubusercontent.com/5157485/27009241-f1f8fe4c-4e3d-11e7-9815-70387b45ee4f.png)

| Locality Variance | Control Variance |
| ----------------- | ---------------- |
| 8.458 |  8.667 |
`T-score: -6.49, P-value: 1.35e-10`

### 4.2. ImageNet
Below are some randomly sampled outputs (not cherry picked) and an analysis of the variances between the two networks. We find that both the input activations and output activations of the LP layer display significantly higher neuronal clustering compared to the control network. 

#### Locality inputs (FC5 outputs)
The left column contains the raw activations, and the right column contains the activations after the mean activations are subtracted.
<img width="873" alt="fc1" src="https://user-images.githubusercontent.com/5157485/27019753-07a546a8-4ef0-11e7-9ca6-430a9ed74df9.png">

| Locality Variance | Control Variance |
| ----------------- | ---------------- |
| 630.9             |  687.4           |
`T-score: -38.98, P-value: 8.087e-203`

#### Locality output activations (FC6 outputs)
<img width="866" alt="local" src="https://user-images.githubusercontent.com/5157485/27019757-0d198edc-4ef0-11e7-9c97-4e67574df290.png">

| Locality Variance | Control Variance |
| ----------------- | ---------------- |
| 653.4             |  691.6           |
`T-score: --26.08, P-value: 8.463e-115`


---

## 5. Conclusions

The Locality Prior layer + weight decay is one way to impose a wiring cost on a network. We show that the LP layer has outputs with significantly lower variance in activations, compared to a control network. We also note that clustering of both inputs and outputs inceases with layer size. 

We think that in a physical network the connections could be sparsified after training. Specifically, the network can be sparsified removing connections which have a small weight in the _W.*P_ matrix. This would be a sort of analogue to reduced brain plasticity in adulthood. We think that this sparsification could be done without a sigificant loss in accuracy.

Another interesting avenue would be to examine whether these representations are nested in the network, as they are in the brain [4].

Finally, the LP layer can be interpreted as the connections between neurons over one timestep. In this light, it would be most effective in a recurrent model. It would be interesting to test the LP layer in a RNN and see if the neurons exhibit stronger functional specification. 

We think that this is a good demonstration that neuron clustering can arise naturally from a wiring cost and network topology. It suggests that functional specification arises naturally from physical constraints and a top-down learning objective. Indeed, it seems fruitful to propose constraints that the brain might be working under and build computational experiments to test these hypotheses, as [6] suggests. We look forward to seeing the results that come out of this marriage of neuroscience and machine learning. 

---

## 6. References

[1] Tosun, D., Rettmann, M. E., Han, X., Tao, X., Xu, C., Resnick, S. M., … Prince, J. L. (2004). Cortical surface segmentation and mapping. NeuroImage, 23(0 1), S108–S118. http://doi.org/10.1016/j.neuroimage.2004.07.042

[2] Kanwisher, Nancy (2010). Functional specificity in the human brain: A window into the functional architecture of the mind. _Proceedings of the National Academy of Sciences, 107_, 11163-11170.
http://www.pnas.org/content/107/25/11163.full.pdf

[3] Haxby, James V., Gobbini, M. Ida, Furey, Maura L., Ishai, Alumit, Schouten, Jennifer L. & Pietrini, Pietro (2001). Distributed and Overlapping Representations of Faces and Objects in Ventral Temporal Cortex. _Science, 293_, 2425-2430.

[4] van den Hurk, Job, Van Baelen, Marc & Op de Beeck, Hans P. (2017). Development of visual category selectivity in ventral visual cortex does not require visual experience. _Proceedings of the National Academy of Sciences, 114_, E4501-E4510.

[5] Hebb, D.O. (1949). The Organization of Behavior. New York: Wiley & Sons.

[6] DiCarlo, J. J., Zoccolan, D., & Rust, N. C. (2012). How does the brain solve visual object recognition? _Neuron, 73(3)_, 415–434. http://doi.org/10.1016/j.neuron.2012.01.010

[7] Yamins, Daniel L. K., Hong, Ha, Cadieu, Charles F., Solomon, Ethan A., Seibert, Darren & DiCarlo, James J. (2014). Performance-optimized hierarchical models predict neural responses in higher visual cortex. _Proceedings of the National Academy of Sciences, 111_, 8619-8624.

---

