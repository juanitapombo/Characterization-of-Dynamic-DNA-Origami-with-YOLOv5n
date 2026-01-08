# Characterization-of-Dynamic-DNA-Origami-with-YOLOv5n
Employing the YOLOv5nano (YOLOv5n) deep neural network (DNN) to automatically identify isolated DNA origami hinge nanostructures from transmission electron microscopy (TEM) images

## 1. IEEE Citation
Y. Wang, X. Jin, and C. Castro, “Accelerating the characterization of dynamic DNA origami
devices with deep neural networks,” Sci. Rep., vol. 13, no. 15196, Sep. 2023. [Online].
Available: https://doi.org/10.1038/s41598-023-41459-w.

## 2. Replication
This replication study focuses on reproducing the “particle detection” component of a deep
learning pipeline proposed by Wang et al. (2023) for the structural analysis of DNA origami
nanodevices. Specifically, our goal is to replicate Figure 2A, 2B, and 2D of the original
work and evaluate the performance of a lightweight object detection model, YOLOv5nano
(YOLOv5n), in localizing individual DNA origami hinge particles within transmission
electron microscopy (TEM) images.
The motivation behind the original work – and by extension, this replication study – is to
automate what is traditionally a tedious and time-consuming process: manually identifying
and measuring dynamic DNA origami nanostructures from TEM micrographs. This manual
approach, although commonly used, limits throughput and introduces potential for human
error. By contrast, a pretrained neural network offers a pathway to rapid, consistent, and
scalable hinge detection, thus dramatically improving efficiency in downstream mechanical
characterization workflows.

#### 2.1 Replication Target
We focused our efforts on reproducing the following results:
- Figure 2A: An example micrograph with predicted bounding boxes drawn around
isolated hinges, illustrating successful particle detection.
- Figure 2B: A performance evaluation of the trained YOLOv5n model, showing F1
score as a function of training image quantity, with an inset plot displaying F1 score
versus confidence threshold across different training sizes.
- Figure 2D: A confusion matrix comparing prediction performance prior to post-
processing. Unlike the original figure, our version does not include the bounding box-
size filter (BBF, Figure 2C) that removes aspect ratio outliers, and thus it only reflects
the model’s raw classification output.

#### 2.2 Replication Method Summary
To replicate these results, we used the pretrained YOLOv5n model implemented in PyTorch
and trained it on the dataset made publicly available by the original authors via the Open
Science Framework (https://doi.org/10.17605/OSF.IO/BMXHF). This dataset includes 50
TEM images of DNA origami hinge nanostructures, each annotated with bounding boxes
in YOLO-compatible format (normalized center x and y coordinates, width, and height).
The images were organized into a test set with 30 micrographs, a validation set with 10
micrographs, and 10 training sets, incrementally increasing in size from 1 to 10 images.
In practice, each training set was evaluated independently to see how model performance
scales with data availability. Performance was then assessed using the F1 score, a balanced
metric combining precision and recall, as depicted in Figure 2B.
Notice that in addition, we explored different training configurations to optimize results.
While the original paper adopted 960 × 960 pixel resolution, we also experimented with
640 × 640 resizing to reduce computational demand. After multiple trials, we found that
using 640 px images with a batch size of 32 offered strong performance that aligned well
with the original findings. Each model was trained 500 epochs under its respective setup.
To conclude, our replication closely mirrors the original study in terms of pipeline structure,
model architecture, as well as evaluation approach. The main difference lies in the absence
of post-processing filters (i.e., the BBF from Figure 2C), meaning that our replication of
Figure 2D is based on unfiltered predictions.

#### 2.3 Replication Results
As mentioned, to assess the effectiveness of our replication, we generated a series of outputs
designed to mirror the key components of Figure 2 from the original study: hinge detection
(2A), model performance as a function of training size and confidence threshold (2B), and
confusion matrix analysis (2D). The results are presented alongside the corresponding
panels from the original publication to enable direct visual and quantitative comparison.
Figure I shows a side-by-side comparison between the original detection output (left) and
our replicated results (right), both overlaid on representative TEM micrographs. The red
bounding boxes denote YOLOv5n-predicted hinge locations, each labeled with class ID and geographical indices. Although the replication outputs contain dense annotations, the
bounding boxes clearly demonstrate the capability of hinge localization consistent with the
original model. In other words, the detection results capture isolated hinges with high visual
fidelity, which validates our reproduction of the “particle detection” central to the pipeline.

![alt text](/figures/Figure1.png)

Figure II compares the original and replicated model performance plots, each displaying
F1 score versus training image count (main panel) and F1 score versus confidence (inset).
Our results are shown for two input resolutions: 960 px (matching the original study) and
640 px (a lower-resolution alternative tested for optimization). Across both configurations,
the F1 score improves with larger training sizes – a trend that follows the original findings.

However, the 640 px model exhibits superior convergence behavior and better alignment
with the original performance peak. Specifically, both our replication and the original study
identify a training set size of 9 images as the point of optimal performance, with F1 scores
approaching 0.8. In contrast, the 960 px model yields more variance and a lower overall
peak, suggesting potential overfitting or reduced efficiency in training under our conditions.
This result highlights the benefit of exploring different resolutions and/or optimized batch
sizes (e.g., 32 in our case) to replicate or even enhance the model efficiency when certain
hyperparameters from the original study are unavailable.

The inset curves, which plot F1 score against confidence threshold, further support the
fidelity of our replication. For models trained on 6-10 images, our curves exhibit the same
overall shape as the original: a sharp initial rise, a plateau near 0.8, and a pronounced drop-
off past a confidence threshold of ~0.6. Although the exact curve shapes vary slightly – as
expected due to inherent randomness in training and lack of full hyperparameter disclosure
– the consistency in peak location, magnitude, and general form has confirmed that our
YOLOv5n implementation successfully reproduced the detection performance dynamics
reported in the original work.

![alt text](/figures/Figure2.png)

![alt text](/figures/Figure3.png)

To assess classification accuracy, we also generated a confusion matrix using the YOLOv5n
model trained on 9 images at 640 px resolution, with results shown in Figure III (right)
alongside the original paper’s matrix (left). Our model produced 305 true positives, 121
false positives, and 67 false negatives. As in the original work, true negatives were not
reported due to the model’s focus on hinge rather than background detection.
Although the raw detection counts differ – potentially due to lower resolution and dataset
variations – the detection trends remain comparable. Notably, the relatively low number of
false negatives suggests strong recall, indicating that our model effectively captured most hinge particles. Furthermore, even our replication omits the step of applying a BBF, the
pre-filtered results demonstrate reasonable agreement with the original’s classification
behavior and thus support the validity of the detection pipeline.

## 3. Extension 

To broaden the scope of the replication study, we proposed a modest extension aimed at
detecting clusters of DNA origami hinges, rather than just the isolated particles. While the
original study exclusively focused on identifying individual hinges for downstream “pose
estimation,” many TEM images contain dense regions where hinges spatially aggregate
and/or overlap. These unrecognized clustered structures can offer key insights into inter-
nanostructure interactions, particularly under varying assembly conditions such as ionic
strength or buffer composition, but were left unexplored in the original pipeline.

#### 3.1 Extension Method Summary

This extension leverages the same set of 50 TEM micrographs from the Open Science
Framework used in the original and replicated experiments (see section 2.2). However, to
transition from isolated particles to clusters, a new layer of manual annotation was required.
These annotations were performed using Roboflow, a widely-employed computer vision
and preprocessing platform that supports object detection workflows. Notably, Roboflow
allows bounding boxes to be delineated over identified features and automatically exports
the annotations in YOLO-compatible format.

Here in our case, we defined clusters as regions where multiple hinges exhibit overlapping
spatial proximity – that is, if hypothetical individual bounding boxes were to intersect or
fall within a shared region, the group was labeled as a cluster. For consistency, all cluster
annotations were performed by a single annotator to avoid subjective variability.

On top of that, to ensure comparability with the original hinge detection setup, we reused
the YOLOv5n architecture and retained the same image resolution (640 px) and batch size
of 32, which had yielded optimal performance in the earlier replication. Only the annotated
label files were modified to reflect the new “cluster” class. This allowed us to test whether
YOLOv5n could also generalize to higher-level object groupings with limited training data,
maintaining the same lightweight deep learning framework as before.

#### 3.2 Extension Results

Figure IV shows an example micrograph demonstrating how hinge clusters were annotated
during the labeling process using Roboflow. These bounding boxes serve as a visualization of how clustered hinges were manually identified for training. To reiterate, clusters were
defined as overlapping or tightly packed hinges that visually co-localize within the TEM
micrographs.

![alt text](/figures/Figure4.png)

The resulting F1 score across training sizes is presented in Figure V (left panel). Compared
to the individual hinge detection model, the cluster-based one displays a lower overall F1
score, peaking around 0.55. This result is expected due to the inherent complexity of cluster
annotations, which capture more diverse and ambiguous features, including variable shapes,
densities, and degrees of overlap between hinges. However, the trend in the learning curve
is smoother and demonstrates superior convergence behavior, indicating that the model is
learning meaningful patterns from the cluster data in a stable manner, even under limited
supervision.

![alt text](/figures/Figure5.png)

The F1-confidence curves, shown in Figure V (right panel), also support this interpretation.
Although the overall performance is weaker than in the replication, the models trained on
9 and 10 images yield prominent peaks, suggesting that the network begins to confidently
distinguish hinge clusters once a sufficient training size is reached. Admittedly, the other
curves appear flatter and noisier, but they still reflect the added challenge of modeling
complex spatial arrangements without explicit sub-structure labels.

These results highlight the value of extending the original detection pipeline to clustered
hinges, which provides a foundation for downstream machine learning tasks. For example,
with clustered regions reliably detected, a logical next step is to perform “pose estimation”
within each bounding box, analogous to the second stage in the original study. However,
angular measurement in clusters introduces new challenges, as overlapping hinges cannot
be easily resolved using standard object detection. A more advanced approach may involve
applying instance segmentation techniques, such as the Mask Region-based Convolutional
Neural Networks (Mask R-CNN), to disentangle individual hinges within a single cluster
label. Alternatively, semi-supervised learning could be leveraged by training models with
limited labeled data (e.g., orientation angles) and unlabeled or weakly labeled examples.
Attention-based architectures could also enhance intra-cluster resolution by focusing the
network’s learning capacity on individual hinge-like substructures, allowing for refined
intra-cluster parsing of poses and orientations.

Overall, this extension builds directly on the original pipeline and expands its applicability
to more realistic and biologically relevant nanostructure configurations. By capturing hinge
interactions and packing behavior, it opens new avenues for studying assembly dynamics
and environmental effects on DNA origami systems.

### 4. Reflection

#### 4.1 Challenges Encountered 

The primary challenge in the replication study stemmed from incomplete documentation
of model training parameters. While the authors specified a 960 px input resolution, key
hyperparameters such as batch size, learning rate, and augmentation settings were omitted.
Given that YOLOv5n is optimized for small datasets, minor variations in these settings can
make performance replication highly sensitive to assumed conditions. Another significant
challenge involved reconstructing F1-confidence curves. The paper did not specify where
confidence scores were stored or how to extract them, so we relied on the legends embedded
in the auto-generated training plots and digitized the curves for comparison across training sizes. This process required intensive manual inspection of logs and visual outputs, many
of which lacked high-resolution export options or accessible raw data. Lastly, although less
critical, retrieving raw detection counts from YOLO’s normalized confusion matrices and
interpreting the paper’s non-standard matrix layout posed additional effort.

#### 4.2 Assumptions and Modifications for Replication

To proceed with the replication, we assumed unspecified training hyperparameters – most
critically, the batch size. After testing several values, we adopted a batch size of 32, which
consistently yielded stable results across training sizes. Additionally, while the original
paper used 960 px input resolution, we modified this to 640 px to improve convergence
and computational efficiency. Surprisingly, this downscaling produced stronger alignment
with the original trends, especially in F1-confidence curves and performance at 9 training
images. For consistency, this 640 px and batch size 32 configuration was carried over to
the extension study. These assumptions and modifications reflect necessary adaptations to
reproduce the results under incomplete methodological disclosure while maintaining the
integrity of the original framework.

#### 4.3 Discrepancies and Their Origins

While our replication successfully reproduced key trends from the original study – such as
peak F1 performance at around 9 training images and the plateauing of F1-confidence
curves at higher training sizes – some quantitative differences remain. Notably, although
our 640 px models outperformed our own 960 px trials, they still yielded slightly lower
F1 scores compared to the original 960 px results. This discrepancy likely arises from the
trade-off in input resolution, which affects the network’s ability to localize finer structural
details in the hinge particles. In the confusion matrix analysis, our true positive and false
positive counts were noticeably lower than those reported in the original study. This is also
likely a consequence of lower resolution and incomplete access to the original training
configuration, which reduce the number of detectable particles overall.

#### 4.4 Recommendations for Improved Replicability
To improve the replicability of the original work, we recommend that the authors release
comprehensive training metadata, particularly the full set of hyperparameters used during
model training, such as the batch size, learning rate, and data augmentation strategies.
Clarifying how confidence thresholding was implemented during evaluation, including the
method for sweeping across confidence levels to generate performance curves, would also
greatly benefit future replication efforts. In addition, making available the raw numerical data underlying plotted results (e.g., F1 score versus confidence curves) would further
enhance transparency and eliminate the need for curve digitization. Finally, documentation
or open-source release of post-processing scripts – such as for confusion matrix generation
and bounding box filtering rules – would help minimize ambiguity and ensure that results
can be reproduced with precision.

#### 4.5 Insights from the Extension
The extension of this study to hinge cluster detection introduced additional complexity but
also yielded promising results. Compared to isolated hinge detection, cluster identification
produced a lower peak F1 score, which is an expected outcome due to greater structural
heterogeneity and ambiguity inherent in cluster annotation. Nevertheless, models trained
on 9 ~ 10 images showed strong convergence and confident predictions, reinforcing the
idea that even minimal manual labeling can support robust YOLO-based performance for
complex object detection tasks. These findings suggest that YOLOv5n can be adapted to
study nanoscale clustering behaviors, which could later be paired with segmentation-based
tools for within-cluster “pose estimation.
” As a continuation of the original pipeline, this
extension broadens the applicability of deep learning in structural DNA nanotechnology.





