# robot/motion.py

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import ARM_MOVE_DURATION_SECONDS, POSE_HOLD_SECONDS, SAFETY_PAUSES
from robot.poses import (
    CUP_START_POSE,
    CUP_APPROACH_POSE,
    CUP_GRASP_POSE,
    CUP_LIFT_POSE,
    CUP_HANDOVER_POSE,
    BASE_POSE,
    CUP_START_POSE,
    CUP_APPROACH_POSE,
    CUP_GRASP_POSE,
    CUP_LIFT_POSE,
    CUP_HANDOVER_POSE,
    MIC_POSE,
    VISITOR_POSE,
    REACHYMIC_POSE,
    RIGHT_67_POSE_1,
    RIGHT_67_POSE_2,
    LEFT_67_POSE_1,
    LEFT_67_POSE_2,
    RIGHT_DISCO_POSE_1,
    RIGHT_DISCO_POSE_2,
    LEFT_DISCO_POSE_1,
    LEFT_DISCO_POSE_2,
    LEFT_ROBOT_DANCE_UP,
    LEFT_ROBOT_DANCE_DOWN,
    RIGHT_ROBOT_DANCE_UP,
    RIGHT_ROBOT_DANCE_DOWN,
    RIGHT_FLOSS_POSE_1,
    RIGHT_FLOSS_POSE_2,
    RIGHT_FLOSS_POSE_3,
    RIGHT_FLOSS_POSE_4,
    LEFT_FLOSS_POSE_1,
    LEFT_FLOSS_POSE_2,
    LEFT_FLOSS_POSE_3,
    LEFT_FLOSS_POSE_4,
    RIGHT_PICTURE_LBERTY_POSE,
    LEFT_PICTURE_LBERTY_POSE,
    LEFT_PICTURE_DAB_POSE,
    RIGHT_PICTURE_DAB_POSE,
    LEFT_PICTURE_YAHOO_POSE,
    RIGHT_PICTURE_YAHOO_POSE,
    RIGHT_PICTURE_YAHOO_INTERMEDIATE_POSE,
)


def wait_if_safety(message):
    if SAFETY_PAUSES:
        input(message)


def validate_pose_matrix(pose):
    pose = np.array(pose, dtype=np.float64)

    if pose.shape != (4, 4):
        raise ValueError("Pose must be a 4x4 matrix.")

    if not np.all(np.isfinite(pose)):
        raise ValueError("Pose contains NaN or infinite values.")

    if not np.allclose(pose[3, :], [0, 0, 0, 1]):
        raise ValueError("Last row must be [0, 0, 0, 1].")

    return pose


def move_4x4(arm, pose, duration=3.0):
    pose = validate_pose_matrix(pose)

    # Keep every recorded arm-pose transition at the configured demo speed.
    duration = ARM_MOVE_DURATION_SECONDS

    print()
    print("Moving to pose:")
    print(np.array2string(pose, precision=4, suppress_small=True))
    print(f"x={pose[0, 3]:.3f}, y={pose[1, 3]:.3f}, z={pose[2, 3]:.3f}")

    arm.goto(
        pose,
        duration=duration,
        wait=True,
    )


def move_both_arms(
    reachy,
    left_pose,
    right_pose,
    duration=None,
    left_intermediate=None,
    right_intermediate=None,
):
    """Move both arms concurrently and wait for both moves to finish."""
    def move_arm(arm, pose):
        if duration is None:
            move_4x4(arm, pose)
            return

        pose = validate_pose_matrix(pose)
        arm.goto(
            pose,
            duration=duration,
            wait=True,
        )

    if left_intermediate is not None or right_intermediate is not None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            via_moves = []
            if left_intermediate is not None:
                via_moves.append(
                    executor.submit(move_arm, reachy.l_arm, left_intermediate)
                )
            if right_intermediate is not None:
                via_moves.append(
                    executor.submit(move_arm, reachy.r_arm, right_intermediate)
                )
            for via_move in via_moves:
                via_move.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        left_move = executor.submit(move_arm, reachy.l_arm, left_pose)
        right_move = executor.submit(move_arm, reachy.r_arm, right_pose)
        left_move.result()
        right_move.result()


def gesture_67(reachy, step_duration=None, total_seconds=None):
    """Perform the crossed, synchronized two-pose 67 gesture."""
    if reachy is None:
        print("Skipping 67 gesture: Reachy not connected.")
        return

    if reachy.l_arm is None or reachy.r_arm is None:
        print("Skipping 67 gesture: both arms are required.")
        return

    print("\nStarting synchronized 67 gesture.")

    pairs = (
        (LEFT_67_POSE_2, RIGHT_67_POSE_1),
        (LEFT_67_POSE_1, RIGHT_67_POSE_2),
    )
    started_at = time.monotonic()
    step = 0

    while True:
        left_pose, right_pose = pairs[step % len(pairs)]
        move_both_arms(
            reachy,
            left_pose=left_pose,
            right_pose=right_pose,
            duration=step_duration,
        )
        step += 1

        if total_seconds is None:
            if step >= len(pairs):
                break
        elif time.monotonic() - started_at >= total_seconds:
            break

    print("67 gesture complete.")


def gesture_disco(reachy, step_duration=1.0, cycles=2):
    """Perform crossed synchronized disco poses for complete cycles."""
    if reachy is None:
        print("Skipping disco gesture: Reachy not connected.")
        return

    if reachy.l_arm is None or reachy.r_arm is None:
        print("Skipping disco gesture: both arms are required.")
        return

    print(f"\nStarting synchronized disco dance ({cycles} cycles).")

    crossed_pairs = (
        # Right pose 1 moves with left pose 2.
        (LEFT_DISCO_POSE_2, RIGHT_DISCO_POSE_1),
        # Right pose 2 moves with left pose 1.
        (LEFT_DISCO_POSE_1, RIGHT_DISCO_POSE_2),
    )

    for _ in range(cycles):
        for left_pose, right_pose in crossed_pairs:
            move_both_arms(
                reachy,
                left_pose=left_pose,
                right_pose=right_pose,
                duration=step_duration,
            )

    print("Disco dance complete.")


def gesture_robot_dance(reachy, step_duration=None, total_seconds=None):
    """Perform the alternating up/down two-pose robot dance."""
    if reachy is None:
        print("Skipping robot dance: Reachy not connected.")
        return

    if reachy.l_arm is None or reachy.r_arm is None:
        print("Skipping robot dance: both arms are required.")
        return

    print("\nStarting synchronized robot dance.")

    pairs = (
        (LEFT_ROBOT_DANCE_UP, RIGHT_ROBOT_DANCE_DOWN),
        (LEFT_ROBOT_DANCE_DOWN, RIGHT_ROBOT_DANCE_UP),
    )
    started_at = time.monotonic()
    step = 0

    while True:
        left_pose, right_pose = pairs[step % len(pairs)]
        move_both_arms(
            reachy,
            left_pose=left_pose,
            right_pose=right_pose,
            duration=step_duration,
        )
        step += 1

        if total_seconds is None:
            if step >= len(pairs):
                break
        elif time.monotonic() - started_at >= total_seconds:
            break

    print("Robot dance complete.")


FLOSS_POSES = (
    (LEFT_FLOSS_POSE_1, RIGHT_FLOSS_POSE_1),
    (LEFT_FLOSS_POSE_2, RIGHT_FLOSS_POSE_2),
    (LEFT_FLOSS_POSE_3, RIGHT_FLOSS_POSE_3),
    (LEFT_FLOSS_POSE_4, RIGHT_FLOSS_POSE_4),
)

# One full floss cycle: positions 1 -> 2 -> 1 -> 3 -> 4 (0-based indices).
FLOSS_CYCLE = (0, 1, 0, 2, 3)


def gesture_floss(reachy, step_duration=None):
    """Perform the synchronized four-pose floss dance."""
    if reachy is None:
        print("Skipping floss gesture: Reachy not connected.")
        return

    if reachy.l_arm is None or reachy.r_arm is None:
        print("Skipping floss gesture: both arms are required.")
        return

    print("\nStarting synchronized floss dance.")

    # Exactly two cycles: 1 -> 2 -> 1 -> 3 -> 4, twice.
    sequence = FLOSS_CYCLE * 2

    for pose_index in sequence:
        left_pose, right_pose = FLOSS_POSES[pose_index]
        move_both_arms(
            reachy,
            left_pose=left_pose,
            right_pose=right_pose,
            duration=step_duration,
        )

    print("Floss dance complete.")


def _fun_picture_pose(
    reachy,
    left_pose,
    right_pose,
    label,
    head_look=None,
    left_intermediate=None,
    right_intermediate=None,
):
    """Move both arms into one synchronized fun-picture pose."""
    if reachy is None:
        print(f"Skipping {label} pose: Reachy not connected.")
        return

    if reachy.l_arm is None or reachy.r_arm is None:
        print(f"Skipping {label} pose: both arms are required.")
        return

    print(f"\nStarting synchronized fun picture pose: {label}.")

    try:
        if head_look is None:
            if left_intermediate is not None or right_intermediate is not None:
                move_both_arms(
                    reachy,
                    left_pose=left_pose,
                    right_pose=right_pose,
                    duration=None,
                    left_intermediate=left_intermediate,
                    right_intermediate=right_intermediate,
                )
            else:
                move_both_arms(
                    reachy,
                    left_pose=left_pose,
                    right_pose=right_pose,
                    duration=None,
                )
        else:
            if left_intermediate is not None or right_intermediate is not None:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    via_moves = []
                    if left_intermediate is not None:
                        via_moves.append(
                            executor.submit(
                                move_4x4,
                                reachy.l_arm,
                                left_intermediate,
                            )
                        )
                    if right_intermediate is not None:
                        via_moves.append(
                            executor.submit(
                                move_4x4,
                                reachy.r_arm,
                                right_intermediate,
                            )
                        )
                    for via_move in via_moves:
                        via_move.result()

            x, y, z, duration = head_look

            def move_head():
                reachy.head.look_at(
                    x=x,
                    y=y,
                    z=z,
                    duration=duration,
                )

            with ThreadPoolExecutor(max_workers=3) as executor:
                head_move = executor.submit(move_head)
                left_move = executor.submit(
                    move_4x4,
                    reachy.l_arm,
                    left_pose,
                )
                right_move = executor.submit(
                    move_4x4,
                    reachy.r_arm,
                    right_pose,
                )
                head_move.result()
                left_move.result()
                right_move.result()
        print(f"{label} pose completed.")
        print(f"Holding the pose for {POSE_HOLD_SECONDS:.0f} seconds.")
        time.sleep(POSE_HOLD_SECONDS)
    except Exception as e:
        print(e)

    


def fun_pose_liberty(reachy, step_duration=None, total_seconds=None):
    """Perform the Statue of Liberty fun picture pose (fnps1)."""
    _fun_picture_pose(
        reachy,
        LEFT_PICTURE_LBERTY_POSE,
        RIGHT_PICTURE_LBERTY_POSE,
        "liberty",
    )


def fun_pose_dab(reachy, step_duration=None, total_seconds=None):
    """Perform the dab fun picture pose (fnps2)."""
    _fun_picture_pose(
        reachy,
        LEFT_PICTURE_DAB_POSE,
        RIGHT_PICTURE_DAB_POSE,
        "dab",
        head_look=(
            0.5,
            0.0,
            -0.12,
            ARM_MOVE_DURATION_SECONDS,
        ),
    )


def fun_pose_yahoo(reachy, step_duration=None, total_seconds=None):
    """Perform the yahoo fun picture pose (fnps3)."""
    _fun_picture_pose(
        reachy,
        LEFT_PICTURE_YAHOO_POSE,
        RIGHT_PICTURE_YAHOO_POSE,
        "yahoo",
        right_intermediate=RIGHT_PICTURE_YAHOO_INTERMEDIATE_POSE,
    )


FUN_PICTURE_POSES = (
    ("liberty", fun_pose_liberty),
    ("dab", fun_pose_dab),
    ("yahoo", fun_pose_yahoo),
)

def open_gripper(arm):
    if arm.gripper is None:
        print("No gripper found.")
        return

    ensure_gripper_on(arm)
    print("Opening gripper...")
    arm.gripper.set_opening(100)
    time.sleep(0.8)


def ensure_gripper_on(arm):
    """Power an arm's gripper before sending an opening command."""
    if arm.gripper is None:
        raise RuntimeError("No gripper is available on this arm.")

    if arm.gripper.is_on():
        return

    print("Gripper is off. Powering on the arm and gripper...")
    arm.turn_on()
    time.sleep(0.5)

    if not arm.gripper.is_on():
        raise RuntimeError("Gripper did not power on.")


def close_gripper_for_ball(arm, opening=25):
    if arm.gripper is None:
        print("No gripper found.")
        return

    ensure_gripper_on(arm)
    print(f"Closing gripper to {opening}...")
    arm.gripper.set_opening(opening)
    time.sleep(1.0)


def look_forward(reachy):
    if reachy is None:
        return

    try:
        reachy.head.look_at(x=0.5, y=0.0, z=0.0, duration=1.0)
    except Exception as e:
        print("look_forward skipped:", e)


def nod(reachy):
    if reachy is None:
        return

    try:
        reachy.head.look_at(x=0.5, y=0.0, z=0.08, duration=0.6)
        reachy.head.look_at(x=0.5, y=0.0, z=-0.06, duration=0.6)
        reachy.head.look_at(x=0.5, y=0.0, z=0.0, duration=0.6)
    except Exception as e:
        print("nod skipped:", e)


def greeting_hand(reachy):
    if reachy is None:
        return

    try:
        if reachy.r_arm.gripper is not None:
            reachy.r_arm.gripper.set_opening(100)
            time.sleep(0.5)
            reachy.r_arm.gripper.set_opening(60)
            time.sleep(0.5)
    except Exception as e:
        print("greeting_hand skipped:", e)


def _antenna_goto(antenna, position, duration=0.4):
    if antenna is None:
        return

    if hasattr(antenna, "goto"):
        antenna.goto(
            position,
            duration=duration,
            wait=False,
            degrees=True,
        )
        return

    if hasattr(antenna, "goal_position"):
        antenna.goal_position = position
        return

    if hasattr(antenna, "joints") and antenna.joints:
        antenna.joints[0].goal_position = position


def set_antennas(reachy, left, right, duration=0.4):
    if reachy is None:
        return

    _antenna_goto(reachy.head.l_antenna, left, duration=duration)
    _antenna_goto(reachy.head.r_antenna, right, duration=duration)


def reset_antennas(reachy):
    set_antennas(reachy, 0.0, 0.0, duration=0.5)


def emotion_antennas(reachy, emotion="happy", stop_event=None):
    if reachy is None:
        return

    patterns = {
        "happy": [
            (12.0, -12.0, 0.15),
            (-12.0, 12.0, 0.15),
        ],
        "confused": [
            (45.0, 10.0, 0.35),
            (10.0, -45.0, 0.35),
            (-30.0, 30.0, 0.35),
        ],
        "sad": [
            (120.0, -120.0, 0.8),
        ],
        "angry": [
            (-55.0, 55.0, 0.25),
            (-35.0, 35.0, 0.25),
        ],
        "neutral": [
            (0.0, 0.0, 0.4),
        ],
    }

    pattern = patterns.get(emotion, patterns["neutral"])

    try:
        index = 0
        while stop_event is None or not stop_event.is_set():
            left, right, duration = pattern[index % len(pattern)]
            set_antennas(reachy, left, right, duration=duration)
            index += 1

            if stop_event is None:
                time.sleep(duration)
                break

            stop_event.wait(duration)

    except Exception as e:
        print("emotion_antennas skipped:", e)


def speaking_motion(reachy, stop_event=None, style="calm"):
    if reachy is None:
        return

    patterns = {
        "greeting": [
            (0.5, -0.08, 0.04, "happy", 0.45),
            (0.5, 0.08, 0.02, "happy", 0.45),
            (0.5, 0.0, 0.06, "happy", 0.45),
            (0.5, 0.0, 0.0, "neutral", 0.45),
        ],
        "thinking": [
            (0.5, 0.05, 0.06, "confused", 0.55),
            (0.5, -0.03, 0.02, "confused", 0.55),
            (0.5, 0.0, 0.0, "neutral", 0.55),
        ],
        "goodbye": [
            (0.5, -0.10, 0.06, "happy", 0.45),
            (0.5, 0.10, 0.03, "happy", 0.45),
            (0.5, -0.08, 0.04, "happy", 0.45),
            (0.5, 0.0, 0.0, "neutral", 0.45),
        ],
        "calm": [
            (0.5, 0.0, 0.05, "neutral", 0.55),
            (0.5, 0.04, 0.0, "happy", 0.55),
            (0.5, -0.04, 0.02, "neutral", 0.55),
            (0.5, 0.0, 0.0, "neutral", 0.55),
        ],
    }

    pattern = patterns.get(style, patterns["calm"])

    try:
        index = 0
        while stop_event is None or not stop_event.is_set():
            x, y, z, emotion, duration = pattern[index % len(pattern)]

            reachy.head.look_at(
                x=x,
                y=y,
                z=z,
                duration=duration,
            )

            emotion_antennas(reachy, emotion=emotion)

            index += 1

            if stop_event is None:
                time.sleep(duration)
                break

            stop_event.wait(duration)

        look_forward(reachy)
        reset_antennas(reachy)

    except Exception as e:
        print("speaking_motion skipped:", e)


def reset_right_arm_to_base(reachy):
    """Return the right arm to the default resting pose."""
    if reachy is None:
        return

    if reachy.r_arm is None:
        print("Right arm not available for reset.")
        return

    print("\nReturning right arm to default position.")
    try:
        move_4x4(reachy.r_arm, BASE_POSE)
    except Exception as e:
        print("Right arm reset skipped:", e)


def reset_after_action(reachy):
    """Restore Reachy's dashboard default posture after an action.

    Use the SDK's native default joint targets while preserving the current
    gripper openings, so the left gripper does not drop the microphone.
    """
    if reachy is None:
        return

    if reachy.l_arm is None or reachy.r_arm is None:
        print("Both arms are required for the post-action reset.")
        return

    print("\nReturning Reachy to the dashboard default posture.")

    try:
        # Cancel any queued action poses before issuing the home posture.
        reachy.cancel_all_goto()

        left_default = reachy.l_arm.get_default_posture_joints("default")
        right_default = reachy.r_arm.get_default_posture_joints("default")

        with ThreadPoolExecutor(max_workers=3) as executor:
            moves = [
                executor.submit(
                    reachy.l_arm.goto,
                    left_default,
                    duration=ARM_MOVE_DURATION_SECONDS,
                    wait=True,
                ),
                executor.submit(
                    reachy.r_arm.goto,
                    right_default,
                    duration=ARM_MOVE_DURATION_SECONDS,
                    wait=True,
                ),
            ]

            if reachy.head is not None:
                moves.append(
                    executor.submit(
                        reachy.head.goto_posture,
                        duration=ARM_MOVE_DURATION_SECONDS,
                        wait=True,
                        wait_for_goto_end=True,
                    )
                )

            for move in moves:
                move.result()

        print("Reachy is back in the dashboard default posture.")
    except Exception as e:
        print("Dashboard default-posture reset skipped:", e)


def return_to_normal_state(reachy, arm=None):
    if reachy is None:
        return

    if arm is None:
        arm = reachy.r_arm

    print()
    print("Returning Reachy to normal state...")

    try:
        if arm is not None:
            open_gripper(arm)
            move_4x4(arm, BASE_POSE, duration=4.0)
    except Exception as e:
        print("Arm normal-state reset skipped:", e)

    look_forward(reachy)
    reset_antennas(reachy)

    print("Reachy is back in normal state.")


def grasp_handover_release(reachy, label="grasp"):
    if reachy is None:
        print(f"Skipping {label}: Reachy not connected.")
        return

    arm = reachy.r_arm

    if arm is None:
        print("Right arm not available.")
        return

    if arm.gripper is None:
        print("Right gripper not available.")
        return

    print()
    print(f"Starting {label} sequence.")
    print("Make sure the cup is in the same position as the recorded poses.")

    wait_if_safety("Press Enter to start movement... ")

    open_gripper(arm)

    move_4x4(arm, CUP_START_POSE, duration=ARM_MOVE_DURATION_SECONDS)
    move_4x4(arm, CUP_APPROACH_POSE, duration=ARM_MOVE_DURATION_SECONDS)

    wait_if_safety("Check CUP_APPROACH_POSE. Press Enter to continue to CUP_GRASP_POSE... ")

    move_4x4(arm, CUP_GRASP_POSE, duration=ARM_MOVE_DURATION_SECONDS)

    wait_if_safety("Check gripper around cup. Press Enter to close gripper... ")

    close_gripper_for_ball(arm, opening=25)

    wait_if_safety("Press Enter to lift... ")

    move_4x4(arm, CUP_LIFT_POSE, duration=ARM_MOVE_DURATION_SECONDS)
    move_4x4(arm, CUP_HANDOVER_POSE, duration=ARM_MOVE_DURATION_SECONDS)

    wait_if_safety("Press Enter to open gripper and release cup... ")

    time.sleep(5.0)

    move_4x4(arm, CUP_LIFT_POSE, duration=ARM_MOVE_DURATION_SECONDS)
    move_4x4(arm, CUP_GRASP_POSE, duration=ARM_MOVE_DURATION_SECONDS)
    open_gripper(arm)
    move_4x4(arm, CUP_APPROACH_POSE, duration=ARM_MOVE_DURATION_SECONDS)
    move_4x4(arm, CUP_START_POSE, duration=ARM_MOVE_DURATION_SECONDS)

    reset_after_action(reachy)

    print(f"{label} sequence complete.")

def grasp_mic_init(reachy, label="mic grab"):
    if reachy is None:
        print(f"Skipping {label}: Reachy not connected.")
        return

    arm = reachy.l_arm

    if arm is None:
        print("Left arm not available.")
        return

    if arm.gripper is None:
        print("Left gripper not available.")
        return

    try:
        # open_gripper(arm)
        move_4x4(arm, MIC_POSE, duration=4.0)
        close_gripper_for_ball(arm, opening=5)
        print(f"{label} Mic grabbed.")
    except RuntimeError as exc:
        # A gripper fault should not terminate the complete voice demo.
        print(f"{label} skipped: {exc}")

def mic_forward(reachy, label="fwd"):
    if reachy is None:
        print(f"Skipping {label}: Reachy not connected.")
        return

    arm = reachy.l_arm

    if arm is None:
        print("Left arm not available.")
        return

    if arm.gripper is None:
        print("Left gripper not available.")
        return

    move_4x4(arm, VISITOR_POSE, duration=4.0)

def mic_backward(reachy, label="bkwd"):
    if reachy is None:
        print(f"Skipping {label}: Reachy not connected.")
        return

    arm = reachy.l_arm

    if arm is None:
        print("Left arm not available.")
        return

    if arm.gripper is None:
        print("Left gripper not available.")
        return

    move_4x4(arm, REACHYMIC_POSE, duration=4.0)
