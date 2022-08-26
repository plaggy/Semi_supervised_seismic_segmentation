# Semi_supervised_seismic_segmentation

The approach was inspired by [1].

The overall workflow is designed to make updates to the existing labels, gradually assigning labels to yet unlabeled pixels. A machine learning model is used to learn representations of classes based on the labels provided, and then probabilities predicted by a pretrained model for unlabeled pixels are used in the label update process. In the experiments, the U-Net model became the backbone of the solution.

The cross pseudo supervision technique [2] was implemented to make the training process more robust. The idea is to initialize two identical models with different weights and use their outputs to supervise each other. The loss for labeled pixels is calculated using labels and model predictions. The additional loss component for unlabeled pixels for each model is calculated with the model’s predicted probabilities and labels obtained from the other model’s probabilities. To make models’ predictions applicable to supervising each other the second loss component was introduced after models were trained in the regular supervised setting for some number of epochs. Thus, the training process became three-staged: at the first stage models were trained using initially labeled pixels only; at the second stage the pseudo-supervised loss component was introduced; the final stage involved making updates to the labels.

The loss function used is confidence-aware focal loss.

After a model is pretrained on existing labels, the labels are being updated on each epoch. The update step is the gist of the procedure and it consists of the following steps:
1.	Class probabilities for each pixel in a partially labeled image are calculated
2.	The neighborhood of existing labels is obtained
3.	Unlabeled pixels are being assigned labels based on model probabilities. The update region is restricted by the neighborhood as well as probability values
4.	The process is repeated until there are no more updates
The updates were only made in a region close to already labeled pixels to preserve spatial continuity and not to allow a model to make predictions in areas far away from what it has already seen. The assumption is that pixels located close to each other are more likely to be similar which will make predictions made on those pixels more meaningful.

[1] Huang, Z.; Wang, X.; Wang, J.; Liu, W.; Wang, J. Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 7014-7023.
[2] Chen, X.; Yuan, Y.; Zeng, G.; Wang, J. Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision. CVPR, 2021
