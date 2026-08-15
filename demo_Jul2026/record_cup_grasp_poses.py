"""Record the five right-arm matrices used by the cup delivery workflow."""

import numpy as np
from reachy2_sdk import ReachySDK

from config import ROBOT_HOST


POSE_STEPS = (
    (
        "CUP_START_POSE",
        "safe starting position before approaching the cup",
    ),
    (
        "CUP_APPROACH_POSE",
        "position close to the cup, but not yet around it",
    ),
    (
        "CUP_GRASP_POSE",
        "final pickup position with the open gripper around the cup",
    ),
    (
        "CUP_LIFT_POSE",
        "cup safely lifted clear of the table or stand",
    ),
    (
        "CUP_HANDOVER_RELEASE_POSE",
        "cup presented to the visitor for handover and release",
    ),
)


def print_pose(name: str, pose) -> None:
    """Print a pose ready to paste into robot/poses.py."""
    matrix = np.asarray(pose, dtype=np.float64)

    print(f"\n{name} = np.array([")
    for row in matrix:
        values = ", ".join(f"{value: .6f}" for value in row)
        print(f"    [{values}],")
    print("], dtype=np.float64)")


def main() -> None:
    reachy = ReachySDK(host=ROBOT_HOST)
    print("Connected:", reachy.is_connected())

    if not reachy.is_connected():
        raise RuntimeError("Cannot connect to Reachy.")

    arm = reachy.r_arm

    if arm is None:
        raise RuntimeError("Right arm is not available.")

    print(
        "\nCUP-GRASP POSE RECORDING MODE\n"
        "1. Support the right arm before making it compliant.\n"
        "2. In the Reachy dashboard, make the RIGHT arm compliant/torque-off.\n"
        "3. Put the cup in its new fixed pickup position.\n"
        "4. Manually position the arm for each requested pose.\n"
        "5. Never force the arm if it feels stiff.\n"
        "6. This script records only; it does not replay movement."
    )

    for step_number, (pose_name, description) in enumerate(
        POSE_STEPS,
        start=1,
    ):
        print(
            f"\nStep {step_number}/{len(POSE_STEPS)}: {pose_name}"
            f"\nPurpose: {description}."
        )
        input(
            f"\nMove the RIGHT arm to {pose_name}, then press Enter... "
        )
        print_pose(pose_name, arm.forward_kinematics())

    print(
        "\nAll five clearly named cup-delivery matrices were recorded. "
        "Copy them into robot/poses.py only after reviewing every value."
    )


if __name__ == "__main__":
    main()
