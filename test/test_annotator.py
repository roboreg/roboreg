import numpy as np
import pytest

from roboreg.annotator import Annotation, OpenCVAnnotator


@pytest.mark.skip(reason="To be fixed.")
def test_opencv_annotator() -> None:
    annotator = OpenCVAnnotator(n_positive=3, n_negative=3)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    annotations = annotator.annotate(img)
    print("Annotations: ", annotations)


@pytest.mark.skip(reason="To be fixed.")
def test_annotation_parser_mixin() -> None:
    annotator = OpenCVAnnotator()
    annotations = [
        Annotation(x=1, y=2, label=1),
        Annotation(x=3, y=4, label=1),
        Annotation(x=5, y=6, label=0),
    ]
    csv_file = "test/assets/samples.csv"
    annotator.write(csv_file, annotations)

    read_annotations = annotator.read(csv_file)
    print("Annotations: ", read_annotations)


if __name__ == "__main__":
    # test_opencv_annotator()
    test_annotation_parser_mixin()
