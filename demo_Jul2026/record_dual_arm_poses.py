"""Record two synchronized Reachy arm-pose pairs and optionally replay them."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from reachy2_sdk import ReachySDK

from config import ARM_MOVE_DURATION_SECONDS, ROBOT_HOST


def print_pose(name: str, pose) -> None:
    """Print a pose in the same format used by robot/poses.py."""
    matrix = np.asarray(pose, dtype=np.float64)

    print(f"\n{name} = np.array([")
    for row in matrix:
        values = ", ".join(f"{value: .6f}" for value in row)
        print(f"    [{values}],")
    print("], dtype=np.float64)")


def capture_arm_pose(reachy, side: str, number: int) -> np.ndarray:
    """Capture one arm's end-effector matrix."""
    arm = reachy.r_arm if side == "right" else reachy.l_arm
    pose_name = f"{side.upper()}_DUAL_POSE_{number}"

    input(
        f"\nPosition the {side.upper()} arm for pose {number}, "
        "then press Enter... "
    )
    pose = np.asarray(
        arm.forward_kinematics(),
        dtype=np.float64,
    )

    print_pose(pose_name, pose)
    return pose


def replay_pair(reachy, left_pose, right_pose, number: int) -> None:
    """Start both arm movements together and wait for both to finish."""
    print(f"\nReplaying dual-arm pose pair {number}...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        left_move = executor.submit(
            reachy.l_arm.goto,
            left_pose,
            duration=ARM_MOVE_DURATION_SECONDS,
            wait=True,
        )
        right_move = executor.submit(
            reachy.r_arm.goto,
            right_pose,
            duration=ARM_MOVE_DURATION_SECONDS,
            wait=True,
        )
        left_move.result()
        right_move.result()


def main() -> None:
    reachy = ReachySDK(host=ROBOT_HOST)
    print("Connected:", reachy.is_connected())

    if not reachy.is_connected():
        raise RuntimeError("Cannot connect to Reachy.")

    if reachy.l_arm is None or reachy.r_arm is None:
        raise RuntimeError("Both Reachy arms must be available.")

    print(
        "\nDUAL-ARM RECORDING MODE\n"
        "1. Do not call turn_on while positioning the arms manually.\n"
        "2. In the Reachy dashboard, make BOTH arms compliant/torque-off.\n"
        "3. Record RIGHT arm poses 1 and 2 first.\n"
        "4. Record LEFT arm poses 1 and 2 afterward.\n"
        "5. Matching pose numbers will replay at the same time.\n"
        "6. Never force an arm that feels stiff."
    )

    right_poses = [
        capture_arm_pose(reachy, "right", number)
        for number in range(1, 3)
    ]
    left_poses = [
        capture_arm_pose(reachy, "left", number)
        for number in range(1, 3)
    ]
    pose_pairs = list(zip(left_poses, right_poses))

    print("\nAll four matrices were recorded above.")

    replay = input(
        "Replay the two synchronized pairs now? Type YES to continue: "
    ).strip()

    if replay != "YES":
        print("Replay skipped.")
        return

    input(
        "Clear the workspace and confirm both arms can move safely. "
        "Press Enter to power the motors and replay... "
    )
    reachy.turn_on()
    # Explicit Arm.turn_on() is required by Reachy SDK 1.0.7 to power each
    # separate gripper after recording in compliant mode.
    reachy.l_arm.turn_on()
    reachy.r_arm.turn_on()

    for number, (left_pose, right_pose) in enumerate(pose_pairs, start=1):
        replay_pair(reachy, left_pose, right_pose, number)

    print("\nDual-arm replay complete.")


if __name__ == "__main__":
    main()
