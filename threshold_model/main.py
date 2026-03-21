import cv2
import numpy as np

from config import (
    MODEL_PATH,
    CAMERA_INDEX,
    WINDOW_NAME,
    MAX_FACES,
    DRAW_LANDMARKS,
    CAM_W,
    CAM_H,
    PANEL_W,
    WINDOW_W,
    WINDOW_H,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_IDX,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    FOREHEAD_IDX,
    CHIN_IDX,
)
from detector import create_face_landmarker, detect_face_landmarks, extract_pixel_landmarks, get_points_by_index
from features import (
    calculate_ear,
    calculate_mar,
    calculate_roll_tilt,
    calculate_pitch,
    extract_patch_with_mask,
    calculate_eye_redness,
)
from tracker import FatigueTracker
from ui import draw_text, draw_points, build_dashboard_panel


def main():
    landmarker = create_face_landmarker(MODEL_PATH, MAX_FACES)
    tracker = FatigueTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    current_ear = 0.0
    current_mar = 0.0
    current_tilt = 0.0
    current_redness = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        result = detect_face_landmarks(frame, landmarker)

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            all_points = extract_pixel_landmarks(face_landmarks, frame_w, frame_h)

            left_eye = get_points_by_index(all_points, LEFT_EYE_IDX)
            right_eye = get_points_by_index(all_points, RIGHT_EYE_IDX)
            mouth = get_points_by_index(all_points, MOUTH_IDX)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0

            current_mar = calculate_mar(mouth)
            current_tilt = calculate_roll_tilt(all_points[LEFT_EYE_OUTER], all_points[RIGHT_EYE_OUTER])
            current_pitch = calculate_pitch(all_points[CHIN_IDX], all_points[FOREHEAD_IDX])

            left_patch, left_mask = extract_patch_with_mask(frame, left_eye)
            right_patch, right_mask = extract_patch_with_mask(frame, right_eye)
            left_redness = calculate_eye_redness(left_patch, left_mask)
            right_redness = calculate_eye_redness(right_patch, right_mask)
            current_redness = (left_redness + right_redness) / 2.0

            tracker.update_frame_metrics(
                ear=current_ear,
                mar=current_mar,
                tilt=current_tilt,
                redness=current_redness,
                pitch=current_pitch,
            )

            if tracker.force_immediate_fatigue:
                tracker.latest_state = "Fatigued"
                tracker.latest_fatigue_score = max(tracker.latest_fatigue_score, 90.0)

            if DRAW_LANDMARKS:
                draw_points(frame, left_eye, (0, 255, 0))
                draw_points(frame, right_eye, (0, 255, 0))
                draw_points(frame, mouth, (255, 0, 0))
                cv2.line(frame, all_points[LEFT_EYE_OUTER], all_points[RIGHT_EYE_OUTER], (0, 255, 255), 2)

            draw_text(frame, f"Live EAR: {current_ear:.3f}", 20, 35, (0, 255, 0), 0.7, 2)
            draw_text(frame, f"Live MAR: {current_mar:.3f}", 20, 70, (0, 255, 0), 0.7, 2)
            draw_text(frame, f"Live Tilt: {current_tilt:.2f}", 20, 105, (0, 255, 0), 0.7, 2)

        else:
            draw_text(frame, "No face detected", 20, 40, (0, 0, 255), 0.9, 2)

        tracker.update_summary_if_needed()

        camera_view = cv2.resize(frame, (CAM_W, CAM_H))

        dashboard = build_dashboard_panel(
            width=PANEL_W,
            height=CAM_H,
            live_ear=current_ear,
            live_mar=current_mar,
            live_tilt=current_tilt,
            live_redness=current_redness,
            avg_ear=tracker.latest_avg_ear,
            avg_mar=tracker.latest_avg_mar,
            avg_tilt=tracker.latest_avg_tilt,
            avg_redness=tracker.latest_avg_redness,
            perclos=tracker.latest_perclos,
            blink_rate=tracker.latest_blink_rate,
            yawn_rate=tracker.latest_yawn_rate,
            nod_rate=tracker.latest_nod_rate,
            continuous_eye_closure_sec=tracker.continuous_eye_closure_sec,
            fatigue_score=tracker.latest_fatigue_score,
            state=tracker.latest_state,
        )

        combined = np.hstack((camera_view, dashboard))
        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()