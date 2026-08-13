from pathlib import Path
from typing import Dict, Tuple, Union

import rich

from roboreg.core.robot import RobotData
from roboreg.io import URDFParser, apply_mesh_origins, load_meshes, simplify_meshes


def _resolve_link_names(
    urdf_parser: URDFParser,
    root_link_name: str,
    end_link_name: str,
    collision: bool,
) -> Tuple[str, str]:
    if root_link_name == "":
        root_link_name = urdf_parser.link_names_with_meshes(collision=collision)[0]
        rich.print(
            f"Root link name not provided. Using the first link with mesh: '{root_link_name}'."
        )
    if end_link_name == "":
        end_link_name = urdf_parser.link_names_with_meshes(collision=collision)[-1]
        rich.print(
            f"End link name not provided. Using the last link with mesh: '{end_link_name}'."
        )
    return root_link_name, end_link_name


def _build_robot_data(
    urdf_parser: URDFParser,
    mesh_paths: Dict[str, Path],
    root_link_name: str,
    end_link_name: str,
    collision: bool,
    target_reduction: float,
) -> RobotData:
    mesh_origins = urdf_parser.mesh_origins(
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )

    # load and preprocess meshes
    meshes = load_meshes(paths=mesh_paths)
    meshes = simplify_meshes(
        meshes=meshes,
        target_reduction=target_reduction,
    )
    meshes = apply_mesh_origins(meshes=meshes, origins=mesh_origins)

    return RobotData(
        meshes=meshes,
        urdf=urdf_parser.urdf,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
    )


def load_robot_data_from_ros_xacro(
    ros_package: str,
    xacro_path: Union[Path, str],
    root_link_name: str = "",
    end_link_name: str = "",
    collision: bool = False,
    target_reduction: float = 0.0,
) -> RobotData:
    r"""Load data to construct a robot from a ROS xacro file.

    Args:
        ros_package (str): ROS package containing the xacro file.
        xacro_path (Union[Path, str]): The xacro path relative to the ros_package.
        root_link_name (str): The root link name of the robot Defaults to the first link with a mesh.
        end_link_name (str): The end link name of the robot Defaults to the last link with a mesh.
        collision (bool): Whether to load collision meshes. Defaults to False.
        target_reduction (float): Mesh simplification in [0, 1]. Defaults to 0.0 (no simplification).

    Returns:
        RobotData: Data for constructing a Robot.
    """
    #  create a URDF parser
    urdf_parser = URDFParser.from_ros_xacro(
        ros_package=ros_package, xacro_path=xacro_path
    )

    root_link_name, end_link_name = _resolve_link_names(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )

    # parse data from URDF
    mesh_paths = urdf_parser.mesh_paths_from_ros_registry(
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )

    return _build_robot_data(
        urdf_parser=urdf_parser,
        mesh_paths=mesh_paths,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
        target_reduction=target_reduction,
    )


def load_robot_data_from_urdf_file(
    urdf_path: Union[Path, str],
    root_link_name: str = "",
    end_link_name: str = "",
    collision: bool = False,
    target_reduction: float = 0.0,
) -> RobotData:
    r"""Load data to construct a robot from a URDF file.

    Args:
        urdf_path (Union[Path, str]): The path to the URDF file. Meshes are resolved relative to this path.
        root_link_name (str): The root link name of the robot Defaults to the first link with a mesh.
        end_link_name (str): The end link name of the robot Defaults to the last link with a mesh.
        collision (bool): Whether to load collision meshes. Defaults to False.
        target_reduction (float): Mesh simplification in [0, 1]. Defaults to 0.0 (no simplification).

    Returns:
        RobotData: Data for constructing a Robot.
    """
    urdf_path = Path(urdf_path)

    #  create a URDF parser
    urdf_parser = URDFParser.from_urdf_file(path=urdf_path)

    root_link_name, end_link_name = _resolve_link_names(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )

    # parse data from URDF
    mesh_uris = urdf_parser.mesh_uris(
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )
    mesh_paths = urdf_parser.resolve_relative_uris(
        uris=mesh_uris, base_path=urdf_path.parent
    )

    return _build_robot_data(
        urdf_parser=urdf_parser,
        mesh_paths=mesh_paths,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
        target_reduction=target_reduction,
    )


def load_robot_data_from_urdf_string(
    urdf: str,
    root_link_name: str = "",
    end_link_name: str = "",
    collision: bool = False,
    target_reduction: float = 0.0,
) -> RobotData:
    r"""Load data to construct a robot from a URDF string. Mesh paths in the URDF must be absolute.

    Args:
        urdf (str): The URDF contents.
        root_link_name (str): The root link name of the robot Defaults to the first link with a mesh.
        end_link_name (str): The end link name of the robot Defaults to the last link with a mesh.
        collision (bool): Whether to load collision meshes. Defaults to False.
        target_reduction (float): Mesh simplification in [0, 1]. Defaults to 0.0 (no simplification).

    Returns:
        RobotData: Data for constructing a Robot.
    """
    urdf_parser = URDFParser(urdf=urdf)

    root_link_name, end_link_name = _resolve_link_names(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )

    # parse data from URDF, expecting absolute mesh paths
    mesh_uris = urdf_parser.mesh_uris(
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )
    mesh_paths = {}
    for link_name, uri in mesh_uris.items():
        mesh_path = Path(uri)
        if not mesh_path.is_absolute():
            raise ValueError(
                f"Expected an absolute mesh path for link '{link_name}', got '{uri}'."
            )
        mesh_paths[link_name] = mesh_path

    return _build_robot_data(
        urdf_parser=urdf_parser,
        mesh_paths=mesh_paths,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
        target_reduction=target_reduction,
    )


def load_robot_data_from_ros_robot_description(
    urdf: str,
    root_link_name: str = "",
    end_link_name: str = "",
    collision: bool = False,
    target_reduction: float = 0.0,
) -> RobotData:
    r"""Load data to construct a robot from a ROS robot_description URDF string (e.g. as
    published by robot_state_publisher). Mesh paths in the URDF must use the package:// prefix,
    resolved via ROS ament_index_python.

    Args:
        urdf (str): The URDF contents.
        root_link_name (str): The root link name of the robot Defaults to the first link with a mesh.
        end_link_name (str): The end link name of the robot Defaults to the last link with a mesh.
        collision (bool): Whether to load collision meshes. Defaults to False.
        target_reduction (float): Mesh simplification in [0, 1]. Defaults to 0.0 (no simplification).

    Returns:
        RobotData: Data for constructing a Robot.
    """
    urdf_parser = URDFParser(urdf=urdf)

    root_link_name, end_link_name = _resolve_link_names(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )

    # parse data from URDF, resolving package:// prefixes via the ROS registry
    mesh_uris = urdf_parser.mesh_uris(
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
    )
    mesh_paths = urdf_parser.resolve_ros_registry_uris(uris=mesh_uris)

    return _build_robot_data(
        urdf_parser=urdf_parser,
        mesh_paths=mesh_paths,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        collision=collision,
        target_reduction=target_reduction,
    )
