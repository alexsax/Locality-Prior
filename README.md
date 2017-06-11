# Locality-Prior

## Intro


## Related Work

## Methods 
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
If the input layer has activations (x) of size K and the output is a vector _y_ of size K activations, then the LP layer has two matrices and a bias vector. The first one, the prior matrix _P_ is KxK and contains the transmitted signal from neuron _i_ to neuron _j_ in element _ij_. The other matrix _W_ is the standard KxK one for a FC layer and the bias _b_ is included here. The prior is multiplied with the activations after the FC layer, elementwise. 

The activations _A_ are $A = P.*(Wx + b)


## Experiments

## Results
Adding a wiring cost to linear layers

## References
[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4587756/
