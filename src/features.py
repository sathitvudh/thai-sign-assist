import numpy as np


NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33


def landmarks_to_array(
    landmark_list,
    expected_points
):

    if landmark_list is None:
        return np.zeros(
            expected_points * 3,
            dtype=np.float32
        )

    values = []

    for landmark in landmark_list.landmark:

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        values,
        dtype=np.float32
    )


def extract_features(results):

    left_hand = landmarks_to_array(
        results.left_hand_landmarks,
        NUM_HAND_LANDMARKS
    )

    right_hand = landmarks_to_array(
        results.right_hand_landmarks,
        NUM_HAND_LANDMARKS
    )

    pose = landmarks_to_array(
        results.pose_landmarks,
        NUM_POSE_LANDMARKS
    )

    return np.concatenate([
        left_hand,
        right_hand,
        pose
    ])
