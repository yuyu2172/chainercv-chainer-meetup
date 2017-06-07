from chainercv.datasets import VOCDetectionDataset
from chainercv.datasets import voc_detection_label_names
from chainercv.visualizations import vis_bbox
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.datasets \
    import voc_semantic_segmentation_label_colors
from chainercv.datasets \
    import voc_semantic_segmentation_label_names
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_label
import matplotlib.pyplot as plot


fig = plot.figure(figsize=(26, 10))
ax1 = fig.add_subplot(1, 2, 1)
plot.axis('off')
ax2 = fig.add_subplot(1, 2, 2)
plot.axis('off')
dataset = VOCDetectionDataset()
img, bbox, label = dataset[310]

vis_bbox(img, bbox, label,
        label_names=voc_detection_label_names,
         ax=ax1)

dataset = VOCSemanticSegmentationDataset()
img, label = dataset[30]
vis_image(img, ax=ax2)
_, legend_handles = vis_label(
    label,
    label_names=voc_semantic_segmentation_label_names,
    label_colors=voc_semantic_segmentation_label_colors,
    alpha=0.9, ax=ax2)
# ax2.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
plot.tight_layout()
plot.savefig('../images/vis_visualization.png')
plot.show()

