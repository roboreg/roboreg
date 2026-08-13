from abc import ABC, abstractmethod
from csv import DictReader, DictWriter
from typing import List, NamedTuple, Tuple

import cv2
import numpy as np
import rich


class Annotation(NamedTuple):
    x: int
    y: int
    label: int  # 1 = positive, 0 = negative


def annotations_to_arrays(
    annotations: List[Annotation],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Split annotations into point coordinate and label arrays, as consumed by a Segmentor.

    Args:
        annotations (List[Annotation]): Annotations to split.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Point coordinates of shape (N, 2) and labels of shape (N,).
    """
    points = np.array([[annotation.x, annotation.y] for annotation in annotations])
    labels = np.array([annotation.label for annotation in annotations])
    return points, labels


class AnnotationParserMixin:
    _fieldnames = ["x", "y", "label"]

    def read(self, path: str) -> List[Annotation]:
        with open(path, "r") as f:
            reader = DictReader(f, fieldnames=self._fieldnames)
            next(reader)  # skip header
            return [
                Annotation(x=int(row["x"]), y=int(row["y"]), label=int(row["label"]))
                for row in reader
            ]

    def write(self, path: str, annotations: List[Annotation]) -> None:
        with open(path, "w") as f:
            writer = DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for annotation in annotations:
                writer.writerow(annotation._asdict())


class Annotator(ABC, AnnotationParserMixin):
    def __init__(self, n_positive: int, n_negative: int) -> None:
        super().__init__()
        self._annotations: List[Annotation] = []
        self._n_positive = n_positive
        self._n_negative = n_negative

    def clear(self) -> None:
        self._annotations = []

    @abstractmethod
    def annotate(self, img: np.ndarray) -> List[Annotation]:
        raise NotImplementedError

    @property
    def annotations(self) -> List[Annotation]:
        return self._annotations

    @property
    def positive_annotations(self) -> List[Annotation]:
        return [annotation for annotation in self._annotations if annotation.label == 1]

    @property
    def negative_annotations(self) -> List[Annotation]:
        return [annotation for annotation in self._annotations if annotation.label == 0]


class OpenCVAnnotator(Annotator):
    def __init__(
        self,
        n_positive: int = 3,
        n_negative: int = 3,
        window_name="Annotate: left click for positive, CTRL + left click for negative samples",
    ) -> None:
        super().__init__(n_positive=n_positive, n_negative=n_negative)
        self._window_name = window_name

    def _on_mouse(self, event, x, y, flags, param):
        if (
            event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY
        ):  # bitwise and for flags: https://stackoverflow.com/questions/32210066/mouse-callback-event-flags-in-python-opencv-osx
            if len(self.negative_annotations) >= self._n_negative:
                rich.print(
                    f"Already added {len(self.negative_annotations)} of {self._n_negative}  negative samples."
                )
                return
            self._annotations.append(Annotation(x=x, y=y, label=0))
            rich.print(
                f"Negative samples: {len(self.negative_annotations)} of {self._n_negative}. Coordinates {x}, {y}."
            )
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.positive_annotations) >= self._n_positive:
                rich.print(
                    f"Already added {len(self.positive_annotations)} of {self._n_positive} positive samples. Use CTRL + Left Click to add negative samples."
                )
                return
            self._annotations.append(Annotation(x=x, y=y, label=1))
            rich.print(
                f"Positive samples: {len(self.positive_annotations)} of {self._n_positive}. Coordinates: {x}, {y}."
            )
            return

    def annotate(self, img: np.ndarray) -> List[Annotation]:
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._on_mouse)
        img_cpy = img.copy()
        while (
            len(self.positive_annotations) < self._n_positive
            or len(self.negative_annotations) < self._n_negative
        ):
            try:
                cv2.imshow(self._window_name, img_cpy)
                cv2.waitKey(10)

                # draw samples
                positive_annotations = self.positive_annotations
                if positive_annotations:
                    last = positive_annotations[-1]
                    cv2.circle(img_cpy, (last.x, last.y), 5, (255, 255, 0), -1)
                negative_annotations = self.negative_annotations
                if negative_annotations:
                    last = negative_annotations[-1]
                    cv2.circle(img_cpy, (last.x, last.y), 5, (0, 255, 255), -1)
            except KeyboardInterrupt:
                break
        cv2.destroyAllWindows()
        return self._annotations
