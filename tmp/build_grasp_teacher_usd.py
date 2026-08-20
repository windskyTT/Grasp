from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Usd, UsdPhysics


DATA_ROOT = Path("/home/windsky/project/Grasp/assets/new_training_set")

OBJECT_NAMES: tuple[str, ...] = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
    "big_tape",
    "blue_pitcher",
    "brush_functional",
    "car_down",
    "cracker_box_oriented",
    "fan_small_head",
    "gun_functional",
    "hammer",
    "loopy_head_side",
    "mouse",
    "off_water_body",
    "small_block",
    "small_tape",
    "solder_iron_head",
    "suger_box_oriented",
    "wood_block_oriented",
)


def prepare_import_urdf(object_name: str, temp_root: Path) -> Path:
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
            f"{object_name}: expected exactly bottom and top links"
        )
    if len(links["top"].findall("collision")) != 1:
        raise RuntimeError(
            f"{object_name}: top must contain exactly one source collision"
        )
    if links["bottom"].findall("collision"):
        raise RuntimeError(
            f"{object_name}: bottom must not contain source collision"
        )

    joints = robot.findall("joint")
    if len(joints) != 1:
        raise RuntimeError(
            f"{object_name}: expected exactly one source joint"
        )
    joint = joints[0]
    if joint.attrib != {"name": "rotation", "type": "revolute"}:
        raise RuntimeError(
            f"{object_name}: source joint must be rotation/revolute"
        )
    if joint.find("parent").attrib["link"] != "bottom":
        raise RuntimeError(
            f"{object_name}: rotation parent must be bottom"
        )
    if joint.find("child").attrib["link"] != "top":
        raise RuntimeError(
            f"{object_name}: rotation child must be top"
        )

    robot.set("name", "TeacherObject")
    for mesh in robot.findall(".//mesh"):
        source_mesh = (object_dir / mesh.attrib["filename"]).resolve()
        mesh.set("filename", str(source_mesh))

    prepared_dir = temp_root / "prepared_urdf"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_urdf = prepared_dir / f"{object_name}.urdf"
    tree.write(prepared_urdf, encoding="utf-8", xml_declaration=True)
    return prepared_urdf


def collision_prims(body_prim: Usd.Prim) -> list[Usd.Prim]:
    return [
        prim
        for prim in Usd.PrimRange(body_prim,Usd.TraverseInstanceProxies(),)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def validate_stage(stage: Usd.Stage, object_name: str) -> None:
    root = stage.GetDefaultPrim()
    if not root or root.GetPath().pathString != "/TeacherObject":
        raise RuntimeError(
            f"{object_name}: default prim must be /TeacherObject"
        )
    if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(
            f"{object_name}: /TeacherObject must be an articulation root"
        )

    bottom = stage.GetPrimAtPath("/TeacherObject/bottom")
    top = stage.GetPrimAtPath("/TeacherObject/top")
    for body_name, body in (("bottom", bottom), ("top", top)):
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"{object_name}: {body_name} must be a rigid body"
            )
        if not body.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError(
                f"{object_name}: {body_name} must retain mass and inertia"
            )

    bottom_colliders = collision_prims(bottom)
    top_colliders = collision_prims(top)
    if bottom_colliders:
        raise RuntimeError(
            f"{object_name}: bottom must remain collision-free"
        )
    if len(top_colliders) != 1:
        raise RuntimeError(
            f"{object_name}: top must contain exactly one collider"
        )

    revolute_joint_prims = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    if len(revolute_joint_prims) != 1:
        raise RuntimeError(
            f"{object_name}: expected exactly one revolute joint"
        )
    joint_prim = revolute_joint_prims[0]
    if joint_prim.GetName() != "rotation":
        raise RuntimeError(
            f"{object_name}: revolute joint must be named rotation"
        )

    joint = UsdPhysics.RevoluteJoint(joint_prim)
    body0_targets = tuple(joint.GetBody0Rel().GetTargets())
    body1_targets = tuple(joint.GetBody1Rel().GetTargets())
    if body0_targets != (bottom.GetPath(),):
        raise RuntimeError(
            f"{object_name}: rotation body0 must target bottom"
        )
    if body1_targets != (top.GetPath(),):
        raise RuntimeError(
            f"{object_name}: rotation body1 must target top"
        )


def convert_to_staging(object_name: str, temp_root: Path) -> Path:
    prepared_urdf = prepare_import_urdf(object_name, temp_root)
    converter_dir = temp_root / "converter" / object_name
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(prepared_urdf),
            usd_dir=str(converter_dir),
            usd_file_name="teacher_object.usd",
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
    validate_stage(converted_stage, object_name)

    staged_dir = temp_root / "flattened" / object_name
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / "teacher_object.usd"
    flattened_layer = converted_stage.Flatten()
    if not flattened_layer.Export(str(staged_path)):
        raise RuntimeError(
            f"{object_name}: failed to export flattened USD"
        )

    staged_stage = Usd.Stage.Open(str(staged_path))
    if staged_stage is None:
        raise RuntimeError(
            f"{object_name}: failed to reopen flattened USD"
        )
    validate_stage(staged_stage, object_name)
    return staged_path


def main() -> None:
    output_paths = {
        object_name: DATA_ROOT / object_name / "teacher_object.usd"
        for object_name in OBJECT_NAMES
    }
    existing_outputs = [
        output_path
        for output_path in output_paths.values()
        if output_path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            f"Teacher USD outputs already exist: {existing_outputs}"
        )

    with tempfile.TemporaryDirectory(
        prefix="grasp_teacher_usd_build_",
        dir="/tmp",
    ) as temp_dir:
        temp_root = Path(temp_dir)
        staged_paths = {
            object_name: convert_to_staging(object_name, temp_root)
            for object_name in OBJECT_NAMES
        }
        for object_name in OBJECT_NAMES:
            output_path = output_paths[object_name]
            shutil.copy2(staged_paths[object_name], output_path)
            print(f"generated {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
