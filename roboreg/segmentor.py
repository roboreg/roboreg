from typing import Any, Union

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Segmentor(object):
    __slots__ = ["_model", "_device"]

    def __init__(self, device: Union[torch.device, str] = "cuda") -> None:
        self._device = torch.device(device) if isinstance(device, str) else device

    def __call__(self, img: np.ndarray) -> Any:
        raise NotImplementedError

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-logits))

    @staticmethod
    def threshold(probability: np.ndarray, pth: float = 0.5) -> np.ndarray:
        return probability > pth


class Sam2Segmentor(Segmentor):
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__(device=device)
        self._model: SAM2ImagePredictor = SAM2ImagePredictor.from_pretrained(
            model_id, device=self._device
        )

    def __call__(
        self, img: np.ndarray, input_points: np.ndarray, input_labels: np.ndarray
    ) -> np.ndarray:
        self._model.set_image(img)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self._device.type, dtype=torch.bfloat16),
        ):
            mask_logits, _, _ = self._model.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
                return_logits=True,
            )
        return self._sigmoid(mask_logits[0])
