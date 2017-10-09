name: inverse
class: center, middle, inverse
layout: true

---

class: titlepage, no-number

# .x-large[ChainerCV: a Library for Deep Learning <br /> in Computer Vision]

.below-60[
## .medium[.author[<u>Yusuke Niitani</u>, Toru Ogawa, Shunta Saito, Masaki Saito ]]]

<!-- <br/> .green[Initial Version: June 18th, 2016]]-->

.below-55[
.left-column40[
.right.img-30[![tokyo logo](images/logo_tokyo.png)]]]
.below-90[
.right-column45[
.left.img-60[![pfn logo](images/pfn_logo.png)]
]]

<!--### .x-small[https://yuyu2172.github.io/chainercv-chainer-meetup]-->

---
layout: false

## ChainerCV

<!--.center.img[![chainercv screenshot](images/screenshot.png)]-->
.center.img-33[![chainercv logo](images/CV1-2.png)]

* Add-on package for CV built on top of Chainer
* Github page:  [https://github.com/chainer/chainercv](https://github.com/chainer/chainercv)
* MIT License
* Developed since late February 2017

---

## Why we developed ChainerCV

.below-60[
#### Make *running* and *training* deep-learning easier in CV
]


<!--
.center.img-90[![](images/example_outputs_first_page_1.png)]
-->


.below-90[
.left-column50[
.center.img-100[![](images/detection.png)]
]
.right-column50[
.center.img-100[![](images/segm.png)]
]
]


---

# Three Guiding Principles

.below-90[.medium[
- **Easy-to-use**
- **Reproduciblity**
- **Compositionality**
]
]


---

template: inverse

# Easy-to-use Implementation

---

## Using other people's research code is hard

.left-column60[
.below-90[
* Unclear API
* Uncommented lines of code
* No installation guidelines
    ]
]
.right-column40[
.center.img-30[
![](images/uncommented_code.png)]
]
<!--* Different implementations have different conventions.-->
<!--* Research code is dirty.-->


---




## ChainerCV is easy-to-use

* Easy installation (`pip install chainercv`)
* Well tested and documented like Chainer
* Unified interface for models (next slide)

<!-- because their instructions are unclear -->

.center.img-50[![](images/readthedoc.png)]

---


## Unified interface for models

#### Object instantiation
```python
FasterRCNNVGG16(pretrained_model='voc07')
SSD300(pretrained_model='voc0712')
```

<!--
```python
FasterRCNNVGG16(pretrained_model='voc07')
SSD300(pretrained_model='voc0712')
SegNet(pretrained_model='camvid')
```
-->

#### Prediction interface
```python
# Detection models
bboxes, labels, scores = model.predict(imgs)
```

<!--
```python
# Semantic Segmentation models
labels = model.predict(imgs)
```
-->

.center.img-80[![](images/detection_api.png)]

---

## Inside `predict` for detection models

.below-60[
1. Preprocesses images
2. Forwards images through network
3. Post-processes outputs
]

<!--
.center.img-75[![](images/predict_doc.png)]
-->
.below-60[
.center.img-100[![](images/inside_predict.png)]
]



---

template: inverse

# Reproducibility

---

## Reproducibility failures

- Different datasets from original paper for eval/train
- Undocumented changes from original paper
- Failure to implement features in original paper


#### When is this problematic?

- Analyzing error
- Extending from current method

.center.img-33[![a](tikz/circle.png)]


---

## ChainerCV for reproducible research

<!--
- Reproduce scores on par with original scores
- Document changes made from original implementation
-->

.below-60[

#### Faster R-CNN

| Training Setting | Evaluation | Reference | ChainerCV |
|:-:|:-:|:-:|:-:|
| VOC 2007 trainval | VOC 2007 test|  69.9 mAP  | **70.5 mAP** |
]


#### SSD300

| Training Setting | Evaluation | Reference | ChainerCV |
|:-:|:-:|:-:|:-:|
| VOC 2007 & 2012 trainval | VOC 2007 test|  77.5 mAP  | **77.5 mAP** |


#### SegNet

| Training Setting | Evaluation | Reference | ChainerCV |
|:--------------:|:---------------:|:--------------:|:----------:|
| CamVid train | CamVid test | 46.3 mIoU | **47.2 mIoU**|

---

template: inverse

# Tools for Training Networks

---

## Overview of training neural network


* `chainer.training` is for general machine learning tasks
* More tools are needed for CV tasks

<!-- Add a slide on how learning a machine software components -->

.center.img[![a](images/software_comp.png)]

---

## Overview of training neural network


* `chainer.training` is for general machine learning tasks
* More tools are needed for CV tasks

<!-- Add a slide on how learning a machine software components -->

.center.img[![a](images/software_comp_thick.png)]

---

## Dataset loader

Similar to dataset loaders in `chainer.datasets` (e.g. MNIST)

```python
from chainercv.datasets import VOCDetectionDataset

*dataset = VOCDetectionDataset(split='trainval', year='2007')
# Access 34th sample in the dataset
img, bbox, label = dataset[34]
```

List of supported datasets:

* PASCAL VOC
* Cityscapes
* CUB-200
* CamVid
* Online Products Dataset

---

## Transform

Modifies an image or an annotation

* `resize`
* `scale`
* `pca_lightning`
* `random_expand` (see below)
* etc...

.center.img-50[![random_expand](images/mnist_random_expand.png)]

---

## TransformDataset

* Extends existing dataset by applying function
* Puts together datasets and transforms

```python
# `dataset` is a dataset for Detection task
def flip_transform(in_data):
    img, bbox, label = in_data
    img, param = random_flip(img, x_flip=True, return_param=True)
    bbox = flip_bbox(bbox, x_flip=param['x_flip'])
    return img, bbox, label

new_dataset = TransformDataset(dataset, flip_transform)
```


.center.img[![kit_fox](images/kit_fox.png)]

<!--
An example where an image is randomly flipped horizontally and bounding box coordinates
are modified based on the flip.
-->

---

## Visualization

* Visualization for images and annotations
  * Images
  * Bounding boxes
  * Segmentation labels
  * Keypoints

* Code is built on top of Matplotlib

.img[![sample_visualization](images/vis_visualization.png)]

---

## Evaluation

Evaluating by standard metric in computer vision

* Semantic Segmentation: IoU
* Object Detection: Average Precision

#### Use as an extension

```python
# trainer is a chainer.training.Trainer object
trainer.extend(
    DetectionVOCEvaluator(iterator, detection_model)
)
```

* The same interface as `Evaluator`

<!-- 
```python
evaluator = chainercv.extension.DetectionVOCEvaluator(
        iterator, detection_model)
# `result` contains dictionary of evaluation results
# ex:  result['main/map'] contains mAP
result = evaluator()
```

-->

---


<!--

## `chainer.training.Extension` for evaluation

```python
# trainer is a chainer.training.Trainer object
trainer.extend(
    chainercv.extension.DetectionVOCEvaluator(iterator, detection_model),
    trigger=(1, 'epoch'))
```

```python
evaluator = chainercv.extension.DetectionVOCEvaluator(
        iterator, detection_model)
# `result` contains dictionary of evaluation results
# ex:  result['main/map'] contains mAP
result = evaluator()
```

Internally, the evaluator runs three operations:

1. Iterate over the iterator to fetch data and make prediction.
2. Pass iterables of predictions and ground truth to `eval_*`.
3. Report results.

-->

<!-- Add a demo if you want to at the first chapter

## `tfdbg`: Screencast and Demo!

.small.right[From Google Brain Team]

<div class="center">
<iframe width="672" height="378" src="https://www.youtube.com/embed/CA7fjRfduOI" frameborder="0" allowfullscreen></iframe>
</div>

<p>

.small[
<br/>
See also: [Debug TensorFlow Models with tfdbg (@Google Developers Blog)](https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html)
]
-->

# Data convention: image

* RGB
* Shape is CHW
* Range of value is [0, 255]


.center.img[![rgb](images/color-channels-RGB.jpg)]

---

# Data convention: bounding box

* Shape is (R, 4)
* `(y_min, x_min, y_max, x_max)` ordered

.center.img-60[![bbox](images/bbox.png)]

```python
bbox == np.array([[150, 100, 400, 600]])
```

---

# Paper

.center.img-80[![ariv](images/arxiv.png)]

* Please cite when using ChainerCV
* Accepted to ACMMM17 Open Source Software Competition

---


## Conclusions

We have talked about the goals of ChainerCV and its solutions.

- Convenient and unified interface to deep learning models
- Baseline implementations for training 
- Training utilities
- Data conventions
- Paper


---
name: last-page
class: center, middle, no-number



<!-- vim: set ft=pandoc -->
