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

