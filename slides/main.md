name: inverse
class: center, middle, inverse
layout: true

---

class: titlepage, no-number

# ChainerCV: a Library for Deep Learning in Computer Vision

## .author[Yusuke Niitani (UTokyo)]
### .small[.white[Jun 10th, 2017]]
<!-- <br/> .green[Initial Version: June 18th, 2016]]-->

### .x-small[https://github.com/chainer/chainercv]


---
layout: false

## ChainerCV

<!--.center.img-50[![chainercv screenshot](images/chainercv/screenshot_large.png)]-->
.center.img[![chainercv screenshot](images/screenshot.png)]

* An add-on package built on top of Chainer for computer vision 
* Github page:  [https://github.com/chainer/chainercv](https://github.com/chainer/chainercv)
* Works with `chainer>=2.0.0`
* MIT License

---

## Goal of ChainerCV
#### Make *running* and *training* deep-learning easier in Computer Vision

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

.center.img[![](images/example_outputs_first_page.png)]

---



## ChainerCV Contributors

* Yusuke Niitani ([@yuyu2172](https://github.com/yuyu2172))
* Toru Ogawa ([@hakuyume](https://github.com/hakuyume))
* Shunta Saito ([@mitmul](https://github.com/mitmul))
* Masaki Saito ([@rezoo](https://github.com/rezoo))
* and more...

---

## Outline of the talk

1. Run-time prototyping
2. Tools for Training Networks
3. Efforts on Reproducible Research
4. Comparison and Conclusions

---

template: inverse

# Run-time prototyping

---

## Using other people's implementation is hard

* Interface is not so clear
* Installation does not work in your environment
* Different implementations have different conventions

.center[
**ChainerCV aims at solving these issues**]

<!-- because their instructions are unclear -->

---


## Unified interface for models

In ChainerCV, using a trained model is as easy as follows.


#### Faster R-CNN
```python
model = FasterRCNNVGG16(pretrained_model='voc07')
bboxes, labels, scores = model.predict(imgs)
```

#### SSD
```python
model = SSD300(pretrained_model='voc0712')
bboxes, labels, scores = model.predict(imgs)
```

#### SegNet
```python
model = SegNet(pretrained_model='camvid')
labels = model.predict(imgs)
```

---

## `predict` for detection models



Internally, `predict` does ...
1. Preprocess images (e.g. mean subtraction and resizing)
2. Forward the images through the network
3. Post-processing outputs by removing overlapping boxes

.center.img-75[![](images/predict_doc.png)]


---

## Potential applications

#### As a building block for other networks

For example, scene graph generation depends
on object detection algorithms to localize objects in images.


.center.img-75[![](images/scene_graph.png)]

---

template: inverse

# Tools for Training Networks

---

## Network training components


* `chainer.training` handles training utilities for general machine learning tasks.
* When applied to concrete problems, additional tools are necessary.

<!-- Add a slide on how learning a machine software components -->

.center.img[![a](images/software_comp.png)]

---

## Network training components


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


List of supported datasets:

* PASCAL VOC
* CUB-200
* CamVid
* Online Products Dataset
* MNIST
* CIFAR 10
* CIFAR 100

---

## Transform

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

## TransformDataset

* An utility to extend an existing dataset by applying a function.
* This puts together datasets and transforms.

```python
from chainercv.datasets import VOCDetectionDataset
from chainer.datasets import TransformDataset
from chainercv.transforms import random_flip, flip_bbox

dataset = VOCDetectionDataset()

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

Evaluating performance of model using standard metric in computer vision.

* Semantic Segmentation: IoU
* Object Detection: Average Precision

<!--These functions are used in evaluation code such as `examples/detection/eval_voc07.py`.-->

---

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


---

template: inverse

# Efforts on Reproducible Research

---
<!-- THIS SLIDED IS NOT REALLY NECESSARY

## Visualization: Example Code

```python
from chainercv.datasets import VOCDetectionDataset
from chainercv.datasets import voc_detection_label_names
from chainercv.visualizations import vis_bbox
import matplotlib.pyplot as plot

*dataset = VOCDetectionDataset()
*img0, bbox0, label0 = dataset[204]
*img1, bbox1, label1 = dataset[700]

fig = plot.figure()

ax1 = fig.add_subplot(1, 2, 1)
plot.axis('off')
*vis_bbox(img0, bbox0, label0,
*        label_names=voc_detection_label_names, ax=ax1)

ax2 = fig.add_subplot(1, 2, 2)
plot.axis('off')
*vis_bbox(img1, bbox1, label1,
*        label_names=voc_detection_label_names, ax=ax2)

plot.show()

```

-->



## Bad implementations *in the wild*

Many reimplementations of algorithms on the internet have problems.

- Trains and evaluates on datasets different from the original paper.
- Undocumented changes from the original implementation.
- Performing X% lower for undocumented reasons and bugs.


---

## ChainerCV for reproducible research

- Reproduce performance on par with the ones reported in the original papers.
- Document changes made from the original implementation.


#### Faster R-CNN

<!--.img[![](images/faster_rcnn_performance.png)]-->

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

## Thank You!
<!--#### [@yuyu2172][Yusuke]-->

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]

<!-- vim: set ft=pandoc -->
