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
2. Forward images through network
3. Post-processing outputs by removing overlapping boxes

.center.img-75[![](images/predict_doc.png)]


---

class: split-40

## Potential applications

#### As a building block for other networks

+ Example: scene graph generation
  + Algorithms depends on detection algorithms
  + Research community focuses on "using" detection results
  + Detection model can be black-box



.center.img-50[![](images/scene_graph.png)]

.small[Scene Graph Generation by Iterative Message Passing. Xu et.al., CVPR2017] 


---

template: inverse

# Efforts on Reproducible Performance

---

## Bad implementations *in the wild*

- Different datasets from original paper for eval/train
- Undocumented changes from original paper
- Failure to implement features in original paper


#### When is this problemetic?

- Analyzing Error
- Extending from the existing method

.center.img-33[![a](tikz/circle.png)]


---

## ChainerCV for reproducible research

- Reproduce scores on par with the original scores
- Document changes made from the original implementation


#### Faster R-CNN

| Training Setting | Evaluation | Reference | ChainerCV | pytorch |
|:-:|:-:|:-:|:-:|
| VOC 2007 trainval | VOC 2007 test|  69.9 mAP  | **70.5 mAP** | 66.1 mAP |


#### SegNet

| Training Setting | Evaluation | Reference | ChainerCV |
|:--------------:|:---------------:|:--------------:|:----------:|
| CamVid train | CamVid test | 46.3 mIoU | **47.2 mIoU**|



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


## Conclusions

We have talked about the goals of ChainerCV and its solutions.

- Convenient and unified interface to deep learning models like Faster R-CNN.
- Support set of baseline implementations for researchers and engineers to extend with new ideas.


---
name: last-page
class: center, middle, no-number



<!-- vim: set ft=pandoc -->
