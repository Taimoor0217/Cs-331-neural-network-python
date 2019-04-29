# Image-Recognition-Neural-Network
Artificial Intelligence , Image Recognition using Neural Networks with Radial Basis Function 
The data set contains link to the Training and Testing Data, which contain the grey values (ranging from 0-255) of images transformed into an array of 784 pixels.
I have used the radial basis function.

The Learned weights get saved with the Name netweights.txt and the
learned features have been saved with the name, netfeatures.txt. The Radial
Basis Function demands the extraction of certain features from the DataSet,
upon which the weights can be learned later on. I have kept the number of
features (clusters) to be 50 for my Neural Network. However, the features
can be increased to improved the overall efficiency of the Network. The
features (also called clusters) of the radial basis function, which can be
simplified to the following
equation h(x)= ∑ exp (−γ ∥ x − xi ∥ ^2) γ ∥ x − xi ∥ ^2) ∥ x − xi ∥ ^2) x −γ ∥ x − xi ∥ ^2) xi ∥ x − xi ∥ ^2) ^2) . Can be
learnt using different ways, I have tried two techniques, both of which can
be found in the code (one commented out). My value of γ ∥ x − xi ∥ ^2) here is
0.00000018. The first one is K-means Clustering and the other is Random
Picking of Clusters. Even though, the former gave a better result, but the
later one also had a very good result and was faster as well, because less
computation needed to be done. So, I decided to use the second method and
increase my Features, so that I can have an even better accuracy. I haveused the Sigmoid function to squish my output values. And have then
computed the error which is the derivative of the sigmoid function. The
derivative of the error function is,
d(E) = output_values*(1 – output_values) * (expected_Values – output_values).
After computing the error, I updated the weights by propagating them backwards to the preceeding layer and the same process
continued for the whole data set of Images.
Accuracy: My algorithm produced an accuracy of 83 % on the training data
and an accuracy of 81.25 % on the test data set.
