import cv2
import numpy as np
import pytest
import torch

from roboreg.annotator import OpenCVAnnotator, annotations_to_arrays
from roboreg.segmentor import Sam2Segmentor


@pytest.mark.skip(reason="To be fixed.")
def test_sam2_segmentor() -> None:
    img = cv2.imread("test/assets/lbr_med7_r800/samples/left_image_1.png")

    # detect
    annotator = OpenCVAnnotator(n_positive=5)
    annotations = annotator.annotate(img)
    points, labels = annotations_to_arrays(annotations)

    # segment
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentor = Sam2Segmentor(device=device)
    p = segmentor(img, points, labels)

    # visualize
    cv2.imshow(
        "masked_img",
        np.where(np.expand_dims(segmentor.threshold(p), -1), img, 0),
    )
    cv2.imshow("probability", p)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_sam2_segmentor()
