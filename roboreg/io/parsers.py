from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import rich
import yaml
from pytorch_kinematics import urdf_parser_py

from roboreg.registration.image.request import CameraObservations, ImageObservations
from roboreg.registration.point_cloud.request import HydraObservations


class URDFParser:
    __slots__ = ["_urdf", "_robot"]

    def __init__(self, urdf: str) -> None:
        self._urdf = urdf
        self._robot = urdf_parser_py.urdf.Robot.from_xml_string(urdf)

    @classmethod
    def from_urdf_file(cls, path: Union[Path, str]) -> "URDFParser":
        r"""Instantiate URDF parser via path to URDF file.

        Args:
            path (Union[Path, str]): Path to URDF file.

        Returns:
            URDFParser: A URDFParser instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"URDF file {path} does not exist.")
        if not path.suffix == ".urdf":
            raise ValueError(f"URDF file {path} must have .urdf extension.")

        with open(path, "r") as f:
            urdf = f.read()

        return cls(urdf=urdf)

    @classmethod
    def from_ros_xacro(
        cls, ros_package: str, xacro_path: Union[Path, str]
    ) -> "URDFParser":
        r"""Instantiate URDF parser from ROS xacro file.

        Args:
            ros_package (str): Internally finds the path to ros_package.
            xacro_path (Union[Path,str]): Path to xacro file relative to ros_package.

        Returns:
            URDFParser: A URDFParser instance.
        """
        xacro_path = Path(xacro_path)
        if not xacro_path.suffix == ".xacro":
            raise ValueError(f"Xacro file {xacro_path} must have .xacro extension.")
        return cls(
            urdf=cls._urdf_from_ros_xacro(
                ros_package=ros_package, xacro_path=xacro_path
            )
        )

    @staticmethod
    def resolve_relative_uris(
        uris: Dict[str, str], base_path: Union[Path, str]
    ) -> Dict[str, Path]:
        r"""Resolve relative URIs using a base path.

        Args:
            uris (Dict[str,str]): Dictionary of link names and relative mesh paths.
            base_path (Union[Path,str]): Base path to resolve relative paths.

        Returns:
            Dict[str,Path]: Dictionary of link names and absolute mesh paths.
        """
        base_path = Path(base_path)
        mesh_paths = {}
        for link_name in uris.keys():
            uri = uris[link_name]
            mesh_paths[link_name] = (base_path / Path(uri)).resolve()
        if len(mesh_paths) != len(uris):
            raise RuntimeError("Some mesh paths could not be resolved.")
        if not all([path.exists() for path in mesh_paths.values()]):
            raise FileNotFoundError("Some resolved mesh paths do not exist.")
        return mesh_paths

    @staticmethod
    def resolve_ros_registry_uris(uris: Dict[str, str]) -> Dict[str, Path]:
        r"""Resolve the URI-style package:// prefix using the ament_index_python ROS registry.

        Args:
            uris (Dict[str,str]): Dictionary of link names and URI-style mesh paths, prefixed with package://.

        Returns:
            Dict[str,Path]: Dictionary of link names and absolute mesh paths.
        """
        from ament_index_python import get_package_share_directory

        mesh_paths = {}
        for link_name in uris.keys():
            uri = uris[link_name]
            if uri.startswith("package://"):
                mesh_path = Path(uri.removeprefix("package://"))
                if len(mesh_path.parts) < 2:
                    raise ValueError(
                        f"Invalid package path {mesh_path} for link {link_name}."
                    )
                mesh_paths[link_name] = (
                    Path(get_package_share_directory(mesh_path.parts[0]))
                    / Path(*mesh_path.parts[1:])
                ).resolve()
            else:
                raise ValueError("Expected a package:// prefix.")
        if len(mesh_paths) != len(uris):
            raise RuntimeError("Some mesh paths could not be resolved.")
        if not all([path.exists() for path in mesh_paths.values()]):
            raise FileNotFoundError("Some resolved mesh paths do not exist.")
        return mesh_paths

    def chain_link_names(self, root_link_name: str, end_link_name: str) -> List[str]:
        r"""Get link names in chain from root to end link.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.

        Returns:
            List[str]: List of link names in chain from root_link_name to end_link_name.
        """
        self._verify_links_in_chain(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        link_names = [root_link_name]
        while link_names[-1] != end_link_name:
            children = self._robot.child_map[link_names[-1]]
            if len(children) != 1:
                raise RuntimeError(f"Expected 1 child, got {len(children)}.")
            _, child_link_name = children[0]
            if link_names[-1] == child_link_name:
                raise RuntimeError(f"End of chain without {end_link_name}.")
            link_names.append(child_link_name)
        return link_names

    def link_names_with_meshes(self, collision: bool = False) -> List[str]:
        r"""Get link names that have meshes.

        Args:
            collision (bool): If True, get collision meshes, else visual meshes.

        Returns:
            List[str]: List of link names with meshes.
        """
        links = [link.name for link in self._robot.links]
        if collision:
            return [link for link in links if self._robot.link_map[link].collision]
        return [link for link in links if self._robot.link_map[link].visual]

    def mesh_uris(
        self, root_link_name: str, end_link_name: str, collision: bool = False
    ) -> Dict[str, str]:
        r"""Get the mesh paths as specified in URDF. These paths may be relative or have a package:// prefix.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.
            collision (bool): If True, get collision mesh paths, else visual mesh paths.

        Returns:
            Dict[str,str]: Dictionary of link names and mesh URIs.
        """
        link_names = self.chain_link_names(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        paths = {}
        # lookup paths
        for link_name in link_names:
            link: urdf_parser_py.urdf.Link = self._robot.link_map[link_name]
            if collision:
                if link.collision is None:
                    continue
                paths[link_name] = link.collision.geometry.filename
            else:
                if link.visual is None:
                    continue
                paths[link_name] = link.visual.geometry.filename
        return paths

    def mesh_paths_from_ros_registry(
        self, root_link_name: str, end_link_name: str, collision: bool = False
    ) -> Dict[str, Path]:
        r"""Get the absolute mesh paths by resolving the package:// prefix using ROS ament_index_python.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.
            collision (bool): If True, get collision mesh paths, else visual mesh paths.

        Returns:
            Dict[str,Path]: Dictionary of link names and absolute mesh paths.
        """
        return URDFParser.resolve_ros_registry_uris(
            uris=self.mesh_uris(
                root_link_name=root_link_name,
                end_link_name=end_link_name,
                collision=collision,
            )
        )

    def mesh_origins(
        self, root_link_name: str, end_link_name: str, collision: bool = False
    ) -> Dict[str, np.ndarray]:
        r"""Get mesh origins.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.
            collision (bool): If True, get collision mesh origins, else visual mesh origins.

        Returns:
            Dict[str,np.ndarray]: Dictionary of link names and mesh origins.
        """
        import transformations

        link_names = self.chain_link_names(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        mesh_origins = {}
        for link_name in link_names:
            link: urdf_parser_py.urdf.Link = self._robot.link_map[link_name]
            if collision:
                if link.collision is None:
                    continue
                link_origin = link.collision.origin
            else:
                if link.visual is None:
                    continue
                link_origin = link.visual.origin
            origin = transformations.euler_matrix(
                link_origin.rpy[0], link_origin.rpy[1], link_origin.rpy[2], "sxyz"
            )
            origin[:3, 3] = link_origin.xyz
            mesh_origins[link_name] = origin
        return mesh_origins

    @staticmethod
    def _urdf_from_ros_xacro(ros_package: str, xacro_path: Path) -> str:
        r"""Convert ROS xacro file to URDF.

        Args:
            ros_package (str): Internally finds the path to ros_package.
            xacro_path (Path): Path to xacro file relative to ros_package.

        Returns:
            str: URDF string.
        """

        import xacro
        from ament_index_python import get_package_share_directory

        path = Path(get_package_share_directory(ros_package)) / xacro_path
        if not path.exists():
            raise FileNotFoundError(f"Xacro file {path} does not exist.")
        return xacro.process(path)

    def _verify_links_in_chain(self, root_link_name: str, end_link_name: str) -> None:
        if not self._robot:
            raise RuntimeError("Robot not initialized.")
        link_names = [link.name for link in self._robot.links]
        if not end_link_name in link_names:
            raise ValueError(f"Link {end_link_name} not in robot.")
        if not root_link_name in link_names:
            raise ValueError(f"Link {root_link_name} not in robot.")

    @property
    def urdf(self) -> str:
        r"""Get URDF string."""
        if self._urdf is None:
            raise ValueError("URDF not loaded.")
        return self._urdf

    @property
    def robot(self) -> urdf_parser_py.urdf.Robot:
        r"""Get robot object."""
        return self._robot


def parse_camera_info(
    camera_info_file: Union[Path, str],
) -> Tuple[int, int, np.ndarray]:
    r"""Parse camera info file.

    Args:
        camera_info_file (Union[Path, str]): Absolute path to the camera info file.

    Returns:
        Tuple[int,int,np.ndarray]:
            - Height of the image.
            - Width of the image.
            - Intrinsic matrix of shape 3x3.
    """
    camera_info_file = Path(camera_info_file)
    with open(camera_info_file, "r") as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    height = camera_info["height"]
    width = camera_info["width"]
    if len(camera_info["k"]) != 9:
        raise ValueError("Camera matrix must be 3x3.")
    intrinsic_matrix = np.array(camera_info["k"]).reshape(3, 3)
    return height, width, intrinsic_matrix


def parse_intrinsics(intrinsics_file: Path | str) -> np.ndarray:
    r"""Parse camera intrinsics from a file.

    Args:
        intrinsics_file: Path to a CSV or YAML file.

    Returns:
        Intrinsic matrix of shape 3x3.
    """
    intrinsics_file = Path(intrinsics_file)
    suffix = intrinsics_file.suffix.lower()
    if suffix == ".csv":
        intrinsics = np.loadtxt(intrinsics_file, delimiter=",")
    elif suffix in {".yaml", ".yml"}:
        with intrinsics_file.open("r") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping in '{intrinsics_file}'.")
        if "intrinsics" in data:
            intrinsics = np.asarray(data["intrinsics"])
        elif "k" in data:
            _, _, intrinsics = parse_camera_info(intrinsics_file)
        else:
            raise ValueError(f"Could not find intrinsics in '{intrinsics_file}'.")
    else:
        raise ValueError(f"Unsupported intrinsics file type '{suffix}'.")
    if intrinsics.size != 9:
        raise ValueError(f"Expected 9 intrinsic values, got {intrinsics.size}.")
    return intrinsics.reshape(3, 3)


def parse_extrinsics(extrinsics_file: Union[Path, str]) -> np.ndarray:
    r"""Parse extrinsics from a NumPy or CSV file.

    Args:
        extrinsics_file (Union[Path, str]): Path to a NumPy or CSV file.

    Returns:
        Extrinsic matrix of shape 4x4.
    """
    extrinsics_file = Path(extrinsics_file)
    suffix = extrinsics_file.suffix.lower()

    if suffix == ".npy":
        extrinsics = np.load(extrinsics_file)
    elif suffix == ".csv":
        extrinsics = np.loadtxt(extrinsics_file, delimiter=",")
    else:
        raise ValueError(
            f"Unsupported extrinsics file type '{suffix}'. "
            "Expected '.npy' or '.csv'."
        )

    if extrinsics.size != 16:
        raise ValueError(f"Expected 16 extrinsic values, got {extrinsics.size}.")

    return extrinsics.reshape(4, 4)


def parse_hydra_observations(
    joint_states_files: List[Path],
    mask_files: List[Path],
    depth_files: List[Path],
) -> HydraObservations:
    r"""Parse data for Hydra registration.

    Args:
        joint_states_files (List[Path]): Joint states files.
        mask_files (List[Path]): Mask files.
        depth_files (List[Path]): Depth files. Note that depth values are expected in meters.

    Returns:
        HydraObservations: Data for Hydra registration.
    """
    if len(joint_states_files) == 0 or len(mask_files) == 0 or len(depth_files) == 0:
        raise ValueError("No files found.")
    if len(joint_states_files) != len(mask_files) or len(joint_states_files) != len(
        depth_files
    ):
        raise ValueError(
            f"Number of files do not match. Got {len(joint_states_files)} joint state files, {len(mask_files)} mask files, and {len(depth_files)} depth files."
        )

    rich.print("Parsing the following files:")
    rich.print(f"Joint states: {[f.name for f in joint_states_files]}")
    rich.print(f"Masks: {[f.name for f in mask_files]}")
    rich.print(f"Depths: {[f.name for f in depth_files]}")

    # load data
    joint_states = [np.load(f) for f in joint_states_files]
    masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]
    depths = [np.load(f) for f in depth_files]
    if not all(
        [mask.shape[:2] == depth.shape[:2] for mask, depth in zip(masks, depths)]
    ):
        raise ValueError("Mask and depth shapes do not match.")
    return HydraObservations(joint_states=joint_states, masks=masks, depths=depths)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to read image '{path}'.")

    return image


def _read_target(path: Path) -> np.ndarray:
    target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if target is None:
        raise ValueError(f"Failed to read target '{path}'.")

    return target


def _validate_image_target_shapes(
    images: list[np.ndarray],
    targets: list[np.ndarray],
    camera_name: str,
) -> None:
    for index, (image, target) in enumerate(zip(images, targets)):
        if image.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Camera '{camera_name}' image and target at index {index} "
                f"have incompatible shapes: {image.shape[:2]} and "
                f"{target.shape[:2]}."
            )


def parse_monocular_observations(
    image_files: list[Path] | None,
    joint_states_files: list[Path],
    target_files: list[Path],
) -> ImageObservations:
    r"""Parse monocular image-registration observations."""

    lengths = {
        "joint_states": len(joint_states_files),
        "targets": len(target_files),
    }

    if image_files is not None:
        lengths["images"] = len(image_files)

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"All observation file lists must have the same length, got {lengths}."
        )

    if not joint_states_files:
        raise ValueError("Expected at least one observation.")

    rich.print("Parsing the following files:")
    if image_files is not None:
        rich.print(f"Images: {[path.name for path in image_files]}")
    rich.print(f"Joint states: {[path.name for path in joint_states_files]}")
    rich.print(f"Targets: {[path.name for path in target_files]}")

    images = (
        [_read_image(path) for path in image_files] if image_files is not None else None
    )
    joint_states = [np.load(path) for path in joint_states_files]
    targets = [_read_target(path) for path in target_files]

    if images is not None:
        _validate_image_target_shapes(
            images=images,
            targets=targets,
            camera_name="camera",
        )

    return ImageObservations(
        joint_states=joint_states,
        cameras={
            "camera": CameraObservations(
                images=images,
                targets=targets,
            )
        },
    )


def parse_stereo_observations(
    left_image_files: list[Path] | None,
    right_image_files: list[Path] | None,
    joint_states_files: list[Path],
    left_target_files: list[Path],
    right_target_files: list[Path],
) -> ImageObservations:
    r"""Parse stereo image-registration observations."""

    lengths = {
        "joint_states": len(joint_states_files),
        "left_targets": len(left_target_files),
        "right_targets": len(right_target_files),
    }

    if left_image_files is not None:
        lengths["left_images"] = len(left_image_files)

    if right_image_files is not None:
        lengths["right_images"] = len(right_image_files)

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"All observation file lists must have the same length, got {lengths}."
        )

    if not joint_states_files:
        raise ValueError("Expected at least one observation.")

    rich.print("Parsing the following files:")
    if left_image_files is not None:
        rich.print(f"Left images: {[path.name for path in left_image_files]}")
    if right_image_files is not None:
        rich.print(f"Right images: {[path.name for path in right_image_files]}")
    rich.print(f"Joint states: {[path.name for path in joint_states_files]}")
    rich.print(f"Left targets: {[path.name for path in left_target_files]}")
    rich.print(f"Right targets: {[path.name for path in right_target_files]}")

    left_images = (
        [_read_image(path) for path in left_image_files]
        if left_image_files is not None
        else None
    )
    right_images = (
        [_read_image(path) for path in right_image_files]
        if right_image_files is not None
        else None
    )

    joint_states = [np.load(path) for path in joint_states_files]
    left_targets = [_read_target(path) for path in left_target_files]
    right_targets = [_read_target(path) for path in right_target_files]

    if left_images is not None:
        _validate_image_target_shapes(
            images=left_images,
            targets=left_targets,
            camera_name="left",
        )

    if right_images is not None:
        _validate_image_target_shapes(
            images=right_images,
            targets=right_targets,
            camera_name="right",
        )

    return ImageObservations(
        joint_states=joint_states,
        cameras={
            "left": CameraObservations(
                images=left_images,
                targets=left_targets,
            ),
            "right": CameraObservations(
                images=right_images,
                targets=right_targets,
            ),
        },
    )
