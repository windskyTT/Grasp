from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Convert the 30 ShapeNet Teacher evaluation URDFs "
        "to standalone bottom and top USDs."
    )
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Usd, UsdPhysics


DATA_ROOT = Path(
    "/home/windsky/project/Grasp/assets/shapenet-30obj"
)

OBJECT_NAMES: tuple[str, ...] = (
    "Bear_34",
    "Black_mug",
    "Blue_camera",
    "Blue_teapot",
    "Camera_brown",
    "Camera_yellow",
    "CellPhone_4e",
    "Donut",
    "DrinkBottle_blue_1ef",
    "Gun",
    "Hammer_40",
    "Knife",
    "Mug_8b_red",
    "Mug_gray_d7",
    "Mug_yellow_18",
    "Pan_gray",
    "Plate_gold_69",
    "Purse_brown_d58",
    "Purse_red_f58",
    "Red_bottle",
    "Red_chair",
    "Red_scissor",
    "Stapler_gray_d9",
    "Teapot_blue_high_3d2",
    "Teapot_brown",
    "Vase_red_a1",
    "Watch_9f",
    "WineGlass_gray_9d",
    "Wine_glass2_1_blue",
    "Wine_glass_body_blue",
)

PART_NAMES: tuple[str, ...] = ("bottom", "top")


def prepare_import_urdf(
    object_name: str,
    part_name: str,
    temp_root: Path,
) -> Path:
    object_dir = DATA_ROOT / object_name
    source_urdf = object_dir / f"{object_name}.urdf"
    tree = ET.parse(source_urdf)
    robot = tree.getroot()

    links = {
        link.attrib["name"]: link
        for link in robot.findall("link")
    }
    if set(links) != {"bottom", "top"}:
        raise RuntimeError(
            f"{object_name}: expected bottom and top links"
        )
    if len(links["top"].findall("collision")) != 1:
        raise RuntimeError(
            f"{object_name}: top must have one source collision"
        )
    if links["bottom"].findall("collision"):
        raise RuntimeError(
            f"{object_name}: bottom must be collision-free"
        )

    joints = robot.findall("joint")
    if len(joints) != 1:
        raise RuntimeError(
            f"{object_name}: expected one source joint"
        )
    joint = joints[0]
    if joint.attrib != {
        "name": "rotation",
        "type": "revolute",
    }:
        raise RuntimeError(
            f"{object_name}: joint must be rotation/revolute"
        )
    if joint.find("parent").attrib["link"] != "bottom":
        raise RuntimeError(
            f"{object_name}: rotation parent must be bottom"
        )
    if joint.find("child").attrib["link"] != "top":
        raise RuntimeError(
            f"{object_name}: rotation child must be top"
        )

    for link_name, link in links.items():
        if link_name != part_name:
            robot.remove(link)
    robot.remove(joint)

    robot.set("name", "TeacherObject")
    for mesh in robot.findall(".//mesh"):
        source_mesh = (
            object_dir / mesh.attrib["filename"]
        ).resolve()
        mesh.set("filename", str(source_mesh))

    prepared_dir = temp_root / "prepared_urdf"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = (
        prepared_dir / f"{object_name}_{part_name}.urdf"
    )
    tree.write(
        prepared_path,
        encoding="utf-8",
        xml_declaration=True,
    )
    return prepared_path


def collision_prims(body_prim: Usd.Prim) -> list[Usd.Prim]:
    return [
        prim
        for prim in Usd.PrimRange(
            body_prim,
            Usd.TraverseInstanceProxies(),
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def validate_stage(
    stage: Usd.Stage,
    object_name: str,
    part_name: str,
) -> None:
    root = stage.GetDefaultPrim()
    if not root or root.GetPath().pathString != "/TeacherObject":
        raise RuntimeError(
            f"{object_name}/{part_name}: "
            "default prim must be /TeacherObject"
        )

    body = stage.GetPrimAtPath(f"/TeacherObject/{part_name}")
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"{object_name}/{part_name}: "
            f"{part_name} must be rigid"
        )
    if not body.HasAPI(UsdPhysics.MassAPI):
        raise RuntimeError(
            f"{object_name}/{part_name}: "
            f"{part_name} must retain mass"
        )

    rigid_bodies = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if rigid_bodies != [body]:
        body_paths = tuple(
            prim.GetPath().pathString
            for prim in rigid_bodies
        )
        raise RuntimeError(
            f"{object_name}/{part_name}: expected only "
            f"{body.GetPath()}, got rigid bodies {body_paths}"
        )

    colliders = collision_prims(body)
    if part_name == "bottom" and colliders:
        raise RuntimeError(
            f"{object_name}/bottom: bottom must remain collision-free"
        )
    if part_name == "top" and len(colliders) != 1:
        raise RuntimeError(
            f"{object_name}/top: top must contain one collider"
        )

    joints = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    if joints:
        raise RuntimeError(
            f"{object_name}/{part_name}: standalone part "
            "must not contain a revolute joint"
        )


def convert_to_staging(
    object_name: str,
    part_name: str,
    temp_root: Path,
) -> Path:
    prepared_urdf = prepare_import_urdf(
        object_name,
        part_name,
        temp_root,
    )
    converter_dir = temp_root / "converter" / object_name / part_name
    usd_file_name = f"{object_name}_{part_name}.usd"
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(prepared_urdf),
            usd_dir=str(converter_dir),
            usd_file_name=usd_file_name,
            fix_base=False,
            merge_fixed_joints=False,
            force_usd_conversion=True,
            joint_drive=None,
            collision_from_visuals=False,
            collider_type="convex_decomposition",
            self_collision=False,
        )
    )
    converted_stage = Usd.Stage.Open(converter.usd_path)
    if converted_stage is None:
        raise RuntimeError(
            f"{object_name}: failed to open converted USD"
        )
    validate_stage(converted_stage, object_name, part_name)

    staged_dir = temp_root / "flattened" / object_name
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / usd_file_name
    flattened_layer = converted_stage.Flatten()
    if not flattened_layer.Export(str(staged_path)):
        raise RuntimeError(
            f"{object_name}: failed to export flattened USD"
        )
    staged_stage = Usd.Stage.Open(str(staged_path))
    if staged_stage is None:
        raise RuntimeError(
            f"{object_name}: failed to reopen staged USD"
        )
    validate_stage(staged_stage, object_name, part_name)
    return staged_path


def main() -> None:
    output_paths = {
        (object_name, part_name): (
            DATA_ROOT
            / object_name
            / f"{object_name}_{part_name}.usd"
        )
        for object_name in OBJECT_NAMES
        for part_name in PART_NAMES
    }
    existing_outputs = [
        path for path in output_paths.values() if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            f"ShapeNet Teacher USD outputs exist: {existing_outputs}"
        )

    with tempfile.TemporaryDirectory(
        prefix="grasp_teacher_shapenet_usd_",
        dir="/tmp",
    ) as temp_dir:
        temp_root = Path(temp_dir)
        staged_paths = {
            (object_name, part_name): convert_to_staging(
                object_name,
                part_name,
                temp_root,
            )
            for object_name in OBJECT_NAMES
            for part_name in PART_NAMES
        }
        for object_name in OBJECT_NAMES:
            for part_name in PART_NAMES:
                output_key = (object_name, part_name)
                output_path = output_paths[output_key]
                shutil.copy2(staged_paths[output_key], output_path)
                print(f"generated {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
