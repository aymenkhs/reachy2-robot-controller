# robot/connection.py

from reachy2_sdk import ReachySDK

from config import ROBOT_HOST, USE_REACHY


def connect_reachy():
    if not USE_REACHY:
        print("USE_REACHY is False. Running without Reachy movement.")
        return None

    reachy = ReachySDK(host=ROBOT_HOST)

    print("Reachy connected:", reachy.is_connected())

    if not reachy.is_connected():
        print("Could not connect to Reachy. Continuing without robot movement.")
        return None

    reachy.turn_on()

    # ReachySDK.turn_on() powers the arm actuators through the generic part
    # API, but in SDK 1.0.7 the separate hand/gripper can remain compliant.
    # Arm.turn_on() explicitly powers its gripper as well.
    for arm_name, arm in (
        ("left", reachy.l_arm),
        ("right", reachy.r_arm),
    ):
        if arm is None:
            continue

        try:
            arm.turn_on()
        except Exception as exc:
            print(f"Could not power the {arm_name} gripper: {exc}")

    try:
        reachy.head.look_at(
            x=0.5,
            y=0.0,
            z=0.0,
            duration=1.0,
        )
    except Exception as e:
        print("Head setup skipped:", e)

    return reachy
