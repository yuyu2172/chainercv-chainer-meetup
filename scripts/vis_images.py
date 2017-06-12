from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plot

import chainer

from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links import SegNetBasic

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset
from chainercv.visualizations import vis_bbox


chainer.config.train = False
dataset = VOCDetectionDataset(year='2007', split='test')
model = SSD512(pretrained_model='voc0712')


indices = [29, 189]

fig = plot.figure(figsize=(30, 60))

for i, idx in enumerate(indices):
    img, _, _ = dataset[idx]
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    ax = fig.add_subplot(2, 2, i + 1)

    vis_bbox(
        img, bbox, label, score,
        label_names=voc_detection_label_names, ax=ax
    )

    # Set MatplotLib parameters
    ax.set_aspect('equal')
    plot.axis('off')
    plot.tight_layout()


plot.show()
