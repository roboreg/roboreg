from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F

from roboreg.core.rendering import NVDiffRastRenderer
from roboreg.core.robot import Robot
from roboreg.core.scene import RobotScene
from roboreg.core.structs import VirtualCamera
from roboreg.registration.image.config import DiffRenderingRegistrationConfig
from roboreg.registration.image.objectives import RenderingObjective
from roboreg.registration.image.request import (
    ImageObservations,
    ImageRegistrationRequest,
)
from roboreg.registration.result import RegistrationResult, TerminationReason
from roboreg.util.transform import rescale_intrinsics


@dataclass(frozen=True)
class OptimizationState:
    iteration: int
    max_iterations: int
    loss: float
    best_loss: float
    learning_rate: float
    extrinsics: torch.Tensor
    renders: dict[str, torch.Tensor]
    camera_losses: dict[str, float]


OptimizationCallback = Callable[[OptimizationState], None]


class DiffRenderingRegistration:
    def __init__(
        self,
        config: DiffRenderingRegistrationConfig,
        objective: RenderingObjective,
        device: torch.device | str = "cuda",
        on_iteration: list[OptimizationCallback] | None = None,
    ) -> None:
        self._config = config
        self._objective = objective
        self._device = torch.device(device)
        self._on_iteration = on_iteration or []

    def __call__(
        self,
        request: ImageRegistrationRequest,
    ) -> RegistrationResult:
        joint_states, preprocessed_targets = self._prepare_image_observations(
            request.observations
        )
        robot_scene = self._create_robot_scene(request)
        extrinsics_9d_inv = self._prepare_extrinsics_9d_inv(request.initial_extrinsics)
        optimizer = self._create_optimizer(params=[extrinsics_9d_inv])
        scheduler = self._create_reduce_on_plateau_scheduler(optimizer)
        result = self._optimize(
            robot_scene=robot_scene,
            joint_states=joint_states,
            preprocessed_targets=preprocessed_targets,
            extrinsics_9d_inv=extrinsics_9d_inv,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return result

    def _prepare_image_observations(
        self,
        observations: ImageObservations,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        joint_states = torch.as_tensor(
            np.stack(observations.joint_states, axis=0),
            dtype=torch.float32,
            device=self._device,
        )
        preprocessed_targets: dict[str, torch.Tensor] = {}
        for camera_name, camera_observations in observations.cameras.items():
            targets = (
                torch.as_tensor(
                    np.stack(camera_observations.targets, axis=0),
                    dtype=torch.float32,
                    device=self._device,
                )
                / 255.0
            )
            # resize on hardware resolution, rendering resolution mismatch
            target_resolution = (
                self._config.camera.target_resolution or camera_observations.shape
            )
            if targets.shape[-2:] != target_resolution:
                targets = F.interpolate(
                    targets.unsqueeze(1),
                    size=target_resolution,
                    mode="nearest",
                ).squeeze(1)
            preprocessed_targets[camera_name] = self._objective.preprocess_targets(
                targets
            )
        return joint_states, preprocessed_targets

    def _create_robot_scene(
        self,
        request: ImageRegistrationRequest,
    ) -> RobotScene:
        # prepare cameras
        cameras: dict[str, VirtualCamera] = {}
        for camera_name, camera_data in request.cameras.items():
            # handle intrinsic scaling: in case of
            # hardware resolution and rendering resolution mismatch
            camera_observations = request.observations.cameras[camera_name]
            native_resolution = camera_observations.shape
            target_resolution = (
                self._config.camera.target_resolution or native_resolution
            )
            intrinsics = rescale_intrinsics(
                intrinsics=camera_data.intrinsics,
                current_resolution=native_resolution,
                target_resolution=target_resolution,
            )
            # prepare virtual camera
            cameras[camera_name] = VirtualCamera(
                resolution=target_resolution,
                intrinsics=intrinsics,
                extrinsics=camera_data.reference_to_camera,
                z_min=self._config.camera.z_min,
                z_max=self._config.camera.z_max,
                device=self._device,
            )
        # prepare robot
        robot = Robot.from_robot_data(
            robot_data=request.robot_data,
            batch_size=len(request.observations.joint_states),
            device=self._device,
        )
        return RobotScene(
            cameras=cameras,
            robot=robot,
            renderer=NVDiffRastRenderer(device=self._device),
        )

    def _prepare_extrinsics_9d_inv(
        self,
        initial_extrinsics: np.ndarray,
    ) -> torch.Tensor:
        extrinsics = torch.as_tensor(
            initial_extrinsics,
            dtype=torch.float32,
            device=self._device,
        )
        # TODO: Standardize transform naming and direction conventions
        # https://github.com/roboreg/roboreg/issues/137
        extrinsics_inv = torch.linalg.inv(extrinsics)
        return (
            pk.matrix44_to_se3_9d(extrinsics_inv).detach().clone().requires_grad_(True)
        )

    def _create_optimizer(
        self, params: Iterable[torch.Tensor]
    ) -> torch.optim.Optimizer:
        return getattr(torch.optim, self._config.optimizer)(params, lr=self._config.lr)

    def _create_reduce_on_plateau_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self._config.plateau_scheduler.mode,
            factor=self._config.plateau_scheduler.factor,
            patience=self._config.plateau_scheduler.patience,
            threshold=self._config.plateau_scheduler.threshold,
        )

    def _optimize(
        self,
        robot_scene: RobotScene,
        joint_states: torch.Tensor,
        preprocessed_targets: dict[str, torch.Tensor],
        extrinsics_9d_inv: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    ) -> RegistrationResult:
        best_extrinsics_inv: torch.Tensor | None = None
        best_loss = float("inf")
        iterations_without_improvement = 0
        for iteration in range(1, self._config.convergence.max_iterations + 1):
            extrinsics_inv = pk.se3_9d_to_matrix44(extrinsics_9d_inv)
            robot_scene.robot.configure(joint_states, extrinsics_inv)
            # per camera render and loss
            camera_losses: dict[str, torch.Tensor] = {}
            renders: dict[str, torch.Tensor] | None = {} if self._on_iteration else None
            for camera_name in robot_scene.cameras:
                render = robot_scene.observe_from(camera_name)
                camera_losses[camera_name] = self._objective(
                    preprocessed_targets=preprocessed_targets[camera_name],
                    renders=render,
                )
                if renders is not None:
                    renders[camera_name] = render
            loss = torch.stack(list(camera_losses.values())).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(metrics=loss)
            loss_value = loss.item()
            if loss_value < best_loss:
                improvement = best_loss - loss_value
                best_loss = loss_value
                best_extrinsics_inv = extrinsics_inv.detach().clone()

                if improvement > self._config.convergence.tolerance:
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            if iterations_without_improvement >= self._config.convergence.patience:
                return RegistrationResult(
                    extrinsics=torch.linalg.inv(best_extrinsics_inv),
                    iterations=iteration,
                    termination_reason=TerminationReason.CONVERGED,
                )
            if self._on_iteration:
                assert renders is not None
                state = OptimizationState(
                    iteration=iteration,
                    max_iterations=self._config.convergence.max_iterations,
                    loss=loss_value,
                    best_loss=best_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    extrinsics=torch.linalg.inv(extrinsics_inv.detach()),
                    renders={
                        camera_name: render.detach()
                        for camera_name, render in renders.items()
                    },
                    camera_losses={
                        camera_name: camera_loss.detach().item()
                        for camera_name, camera_loss in camera_losses.items()
                    },
                )
                for callback in self._on_iteration:
                    callback(state)
        return RegistrationResult(
            extrinsics=torch.linalg.inv(best_extrinsics_inv),
            iterations=iteration,
            termination_reason=TerminationReason.MAX_ITERATIONS,
        )
