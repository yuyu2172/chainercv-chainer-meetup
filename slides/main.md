name: inverse
class: center, middle, inverse
layout: true

---

class: titlepage, no-number

# ChainerCV: Library for Deep Learning in Computer Vision

## .author[Yusuke Niitani (UTokyo)]
### .small[.white[Jun 10th, 2017]]
<!-- <br/> .green[Initial Version: June 18th, 2016]]-->

### .x-small[https://yuyu2172.github.io/chainercv-chainer-meetup]


---
layout: false

## ChainerCV

.center.img[![chainercv screenshot](images/screenshot.png)]

* Add-on package for CV built on top of Chainer
* Github page:  [https://github.com/chainer/chainercv](https://github.com/chainer/chainercv)
* MIT License
* Developed since late February 2017

---

## ChainerCV Contributors

* Yusuke Niitani ([@yuyu2172](https://github.com/yuyu2172))
* Toru Ogawa ([@hakuyume](https://github.com/hakuyume))
* Shunta Saito ([@mitmul](https://github.com/mitmul))
* Masaki Saito ([@rezoo](https://github.com/rezoo))
* and more...

---

## Why do we develop ChainerCV
#### Make *running* and *training* deep-learning easier in CV

<!--
* Network implementations and training scripts
  * Object Detection (Faster R-CNN, SSD)
  * Semantic Segmentation (SegNet)
* Variety of tool sets 
* Dataset Loader (e.g. PASCAL VOC) and data-augmentation tools
* Visualization
* Evaluation
-->

<!--.center.img-33[![Right-algined text](images/faster_rcnn_image_000008.png)]-->

.center.img-90[![](images/example_outputs_first_page_1.png)]

---

## Outline of the talk

1. Easy-to-use Implementation
2. Tools for Training Networks
3. Efforts on Reproducible Research
4. Comparison and Conclusions

---

template: inverse

# Easy-to-use Implementation

---

## Using other people's research code is hard

* Interface is not so clear (e.g. How to run with my data?)
* Installation failure (e.g. Unmaintained Caffe)
<!--* Different implementations have different conventions.-->
<!--* Research code is dirty.-->

--

.below-60[.center[
**ChainerCV aims at solving these issues**]]

1. Easy installation (`pip install chainercv`)
2. Well tested and documented like Chainer
3. Unified interface (next slide)

<!-- because their instructions are unclear -->

---


## Unified interface for models


#### Object instantiation
```python
FasterRCNNVGG16(pretrained_model='voc07')
SSD300(pretrained_model='voc0712')
SegNet(pretrained_model='camvid')
```

#### Prediction interface
```python
# Detection models
bboxes, labels, scores = model.predict(imgs)
```

```python
# Semantic Segmentation models
labels = model.predict(imgs)
```

---

## Inside of `predict` for detection models

Internally, `predict` does ...
1. Preprocess images (e.g. mean subtraction and resizing)
2. Forward the images through the network
3. Post-processing outputs by removing overlapping boxes

.center.img-75[![](images/predict_doc.png)]


---

class: split-40

## Potential applications

#### As a building block for other networks

+ Example: scene graph generation
  + Algorithms depends on detection algorithms
  + Research community focuses on how to use detection results
  + Detection model can be black-box



.center.img-50[![](images/scene_graph.png)]

.small[Scene Graph Generation by Iterative Message Passing. Xu et.al., CVPR2017] 

---

template: inverse

# Tools for Training Networks

---

## Overview of training neural network


* `chainer.training` handles training utilities for general machine learning tasks.
* When applied to concrete problems, additional tools are necessary.

<!-- Add a slide on how learning a machine software components -->

.center.img[![a](images/software_comp.png)]

---

## Overview of training neural network


* `chainer.training` handles training utilities for general machine learning tasks.
* When applied to concrete problems, additional tools are necessary.

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


#### List of supported datasets

* PASCAL VOC
* CUB-200
* CamVid
* Online Products Dataset
* MNIST
* CIFAR 10
* CIFAR 100

---

## Data Preprocessing: Transform

A function that takes an image and annotations as inputs and applies a modification to the inputs

* Transforms for images
    * `resize`
    * `scale`
    * `pca_lightning`
    * `random_expand` (see below)
    * etc...
* Transforms for annotations such as bounding boxes and keypoints

.center.img-50[![random_expand](images/mnist_random_expand.png)]

---

## Data Preprocessing: TransformDataset

* An utility to extend an existing dataset by applying a function.
* This puts together datasets and transforms.

```python
# `dataset` is a dataset for Detection task
def flip_transform(in_data):
    img, bbox, label = in_data
    img, param = random_flip(img, x_flip=True, return_param=True)
    bbox = flip_bbox(bbox, x_flip=param['x_flip'])
    return img, bbox, label

new_dataset = TransformDataset(dataset, flip_transform)
```

An example where an image is randomly flipped horizontally and bounding box coordinates
are modified based on the flip.

---

## Visualization

* Visualization for all the data types used in ChainerCV
  * Images
  * Bounding boxes
  * Segmentation labels
  * Keypoints

* Code is built on top of Matplotlib

.img[![sample_visualization](images/vis_visualization.png)]

---

## Evaluation

Evaluating performance of models using standard metric in computer vision.

* Semantic Segmentation: IoU
* Object Detection: Average Precision

#### `chainer.training.Extension` for evaluation

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

---

template: inverse

# Efforts on Reproducible Research

---

## Bad implementations *in the wild*

- Trains and evaluates on datasets different from the original paper.
- Undocumented changes from the original implementation.

This is problemetic for developing and comparing new ideas to the existing ones.

---

## ChainerCV for reproducible research

- Reproduce performance on par with the ones reported in the original papers.
- Document changes made from the original implementation.


#### Faster R-CNN

| Training Setting | Evaluation | Reference | ChainerCV |
|:-:|:-:|:-:|:-:|
| VOC 2007 trainval | VOC 2007 test|  69.9 mAP  | **70.5 mAP** |


#### SegNet

| Training Setting | Evaluation | Reference | ChainerCV |
|:--------------:|:---------------:|:--------------:|:----------:|
| CamVid train | CamVid test | 46.3 mIoU | **47.2 mIoU**|



---

template: inverse

# Comparison and Conclusions

---

## Comparison of deep learning libraries in Computer Vision

|   | ChainerCV *  | pytorch/vision     |
|---|---|---|---|
| **Backend** | Chainer | PyTorch |
| **Supported Models** | Classification, detection and semantic segmentation  | Classification models |
| **# of Transforms** | 17  | 11 |
| **Visualization** | .blue[Y]  | .red[N] |
| **Evaluation** | .blue[Y] | .red[N] |

*Combination of ChainerCV and vision related functionalities in Chainer.

This comparison is valid as of June 10th, 2017.

---

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


## Concluding Remarks

We have talked about the goals of ChainerCV and its solutions.

- Convenient and unified interface to deep learning models like Faster R-CNN.
- Set of tools to train a model in computer vision.
- Support set of baseline implementations for researchers and engineers to extend with new ideas.


---
name: last-page
class: center, middle, no-number



<!-- vim: set ft=pandoc -->
