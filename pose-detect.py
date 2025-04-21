import cv2
import mediapipe as mp
import math

# Metrics storage
alignment_scores = []
stance_scores    = []
sym_scores       = []
jerk_vals        = []
prev_vel         = None
prev_com         = None

def calc_angle(a, b, c):
    # Compute acute angle ABC
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba)*math.hypot(*bc) + 1e-6)
    deg = math.degrees(math.acos(max(-1, min(1, cos_angle))))
    return deg if deg <= 90 else 180 - deg

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Video I/O setup
cap    = cv2.VideoCapture('form-part1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out    = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

            h, w = frame.shape[:2]
            lm   = results.pose_landmarks.landmark

            def pt(l): return (l.x * w, l.y * h)

            # Keypoints
            l_sh = pt(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            r_sh = pt(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            l_hp = pt(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
            r_hp = pt(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
            l_kn = pt(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
            r_kn = pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            l_an = pt(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            r_an = pt(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

            # 1) Alignment score
            angle_sh_hp = calc_angle(l_sh, r_sh, r_hp)
            align_score = max(0, 100 - (angle_sh_hp / 90) * 100)
            alignment_scores.append(align_score)

            # 2) Stance depth score
            knee_angle   = calc_angle(l_hp, l_kn, l_an)
            stance_score = max(0, 100 - abs(knee_angle - 160) / 20 * 100)
            stance_scores.append(stance_score)

            # 3) Symmetry score
            knee_angle_r = calc_angle(r_hp, r_kn, r_an)
            sym_score    = max(0, 100 - abs(knee_angle - knee_angle_r) / 20 * 100)
            sym_scores.append(sym_score)

            # 4) Smoothness score
            com = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
            if prev_com is not None:
                vel = (com[0] - prev_com[0], com[1] - prev_com[1])
                if prev_vel is not None:
                    jerk = math.hypot(vel[0] - prev_vel[0], vel[1] - prev_vel[1])
                    jerk_vals.append(jerk)
                prev_vel = vel
            prev_com = com

        # Compute running averages every frame
        if alignment_scores:
            avg_align   = sum(alignment_scores) / len(alignment_scores)
            avg_stance  = sum(stance_scores)    / len(stance_scores)
            avg_sym     = sum(sym_scores)       / len(sym_scores)
            avg_jerk    = sum(jerk_vals)        / len(jerk_vals) if jerk_vals else 0
            smooth_score= max(0, 100 - avg_jerk / 50 * 100)
            overall     = (avg_align + avg_stance + avg_sym + smooth_score) / 4

            # Prepare overlay texts
            texts = [
                f"Alignment: {avg_align:.0f}%",
                f"Stance:    {avg_stance:.0f}%",
                f"Symmetry:  {avg_sym:.0f}%",
                f"Smooth:    {smooth_score:.0f}%",
                f"Overall:   {overall:.0f}%"
            ]

            # Overlay in top-right corner
            padding, line_height = 10, 30
            x_base = w - padding
            y = padding + line_height // 2
            for txt in texts:
                (text_w, text_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x = x_base - text_w
                cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
                cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                y += line_height

        out.write(frame)

# Release resources
cap.release()
out.release()
