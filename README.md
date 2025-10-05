# Masterthesis

Public repo with my masters in Computational Science: Physics at University of Oslo, spring 2023.

The code and the thesis is provided as-is.

## Supervisors

- Anders Malthe-Sørensen
- Mikkel Lipperød
- Konstantin Holzhausen

## Abstract

This thesis explores the use of a local learning rule to train the filters of a convolution neural network, to determine if this leads to more robust models than those trained with the famous backpropagation algorithm. While machine learning and neural networks have become im- portant statistical tools, capable of finding complex patterns in large amounts of data, they are also highly vulnerable to adversarial attacks. These attacks are small perturbations to the input data, aimed to fool the model into making incorrect predictions. This is a problem for the use of neural networks in safety-critical applications and is important to understand.

The update rule for the parameters of the network is based on local information, and takes inspiration from biological learning, though it does not aim to be biologically plausible. The neural network I used was a feed-forward network, and related to dense associative memory- models. This relation restricts the architecture I used to models with a single hidden layer. I trained a large number of models, exploring the effects of the various hyperparameters. The best performing models had accuracies of 72%, both for the locally trained models, and the backpropagation-trained models I used to compare them against. I defined a quantitative measure for the robustness of the models, in order to make direct comparisons between mod- els. I found there to be a trade-off where models with high accuracy have low robustness. By training ensembles of models with the same hyperparameters I was able to isolate the effect of each, and find that many of the hyperparameters resulted in this trade-off. This was not seen in the backpropagation-trained models.

While my quantitative definition of robustness fails to fully capture the complex nature of ro- bustness, I can still satisfyingly compare the models and draw concolusions. The behaviour of the robustness of all the models was the same for various ways of measuring the robustness. This was true both when using Fast Gradient Sign Method and the stronger attack, Projected Gradient Descent.

Machine learning models are often described to be black boxes, where the patterns they learn are not understandable from the outside. I found that the locally trained models had much smoother convolution filters that could easily be interpreted as colours or edges, something not seen in the backpropagation-trained models. I also showed while not all of the filters were smooth, the models with the highest robustness had a smaller fraction of these unconverged filters.
