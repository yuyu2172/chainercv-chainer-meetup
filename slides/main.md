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


## Why we developed ChainerCV

.below-60[
#### Make *running* and *training* deep-learning easier in CV
]


<!--
.center.img-90[![](images/example_outputs_first_page_1.png)]
-->


.below-90[
.left-column50[
Supported tasks
- Object detection
- Semantic segmentation
- Image classification

]
.right-column50[
.center.img-100[![](images/detection.png)]
.center.img-100[![](images/segm.png)]
]
]



---

# Three guiding principles

.below-90[.medium[
- **Ease of use**
- **Reproduciblity**
- **Compositionality**
]
]


---

template: inverse

# Ease of use

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

.center.img-60[![](images/detection_api.png)]

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


.below-60[
.center.img-90[![a](images/devils_in_detail.jpg)]]


[comment]:  (Original image from the following link. I added text. https://www.flickr.com/photos/amelien/26643253432/in/photolist-GAnD1u-Wgma5S-61NZ9W-eJbya1-HKvYhv-eE5Qs7-r9PW4k-NFcNb-VuDFJN-dehD6U-qi5b7y-A1os3-SKkWF8-9wD9Uf-eaYbJc-4kRd5n-5Dbf8n-8fCZTg-mzN7H-czDtGo-4ejiyx-JSddy-ev3aUp-UHENRo-hdcJ5n-o5CDHK-nQoD-X2GPoz-5JkD5T-5RuRbA-EBGxGZ-VNBai3-8y2JSW-eRMooE-81t36-VYsY7a-3dTbSr-eJjimF-dvsyW2-6fL73J-8wQnsS-SRcLK8-bKj2t4-LTayWC-8CzkXC-6eDkqm-9gPZva-n7FyDt-hxd6xE-7MKtWv)

---


## Why reproducibility is important

- Analyzing error
- Extending from current method

.below-60[
.center.img-60[![a](tikz/circle.png)]
]


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

# Compositionality

---

## Abstraction for Utilities

.below-45[
* Decompose training process into several blocks
* The format of data is consistent
]


<!-- Add a slide on how learning a machine software components -->

<!--
.below-60[
.center.img[![a](images/software_comp.png)]
]
-->

.below-20[
.center.img-75[![a](images/overview_util.png)]
]

---

# Demo


---

## Conclusions

.below-60[
ChainerCV ...
]
- provides **easy-to-use** implementations of sophisticated networks like SSD
- faithfully **reproduced** training procedure from the original papers
- contains tools and implementations that are **compositional**

.center.img-33[![chainercv logo](images/CV1-2.png)]
.center[Install by  `pip install chainercv`]


---
name: last-page
class: center, middle, no-number

---

<!-- vim: set ft=pandoc -->
