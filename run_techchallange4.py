import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import os
from deepface import DeepFace
from mediapipe.framework.formats import landmark_pb2
from collections import defaultdict
from datetime import datetime # Import datetime for current date/time

# --- Configurações e Constantes Globais ---
MIN_LANDMARK_VISIBILITY = 0.1
# FACE_CROP_RATIO_Y e FACE_CROP_RATIO_X foram removidos pois não são mais utilizados.

# Limiares para detecção de postura (ajustáveis)
THRESHOLD_LYING_ASPECT_RATIO = 1.8
THRESHOLD_LYING_VERTICAL_HEIGHT_RATIO = 0.4

SITTING_BBOX_HEIGHT_MIN_RATIO = 0.15
SITTING_KNEE_HIP_MAX_DIST_RATIO = 0.10
SITTING_ANKLE_HIP_MAX_DIST_RATIO = 0.35
ANGLE_KNEE_SITTING_MIN = 60
ANGLE_KNEE_SITTING_MAX = 120
ANGLE_HIP_SITTING_MIN = 60
ANGLE_HIP_SITTING_MAX = 120

STANDING_LEG_TO_BBOX_RATIO_MIN = 0.40
STANDING_HIP_KNEE_TO_BBOX_RATIO_MIN = 0.20
ANGLE_KNEE_STANDING_MIN = 20
ANGLE_KNEE_STANDING_MAX = 30
ANGLE_HIP_STANDING_MIN = 0
ANGLE_HIP_STANDING_MAX = 30

# Limiares para Movimento de Membros (Nova função)
MOVEMENT_THRESHOLD_MEMBERS = 5

# --- Dicionários de Histórico Globais para 'detect_activities' e 'analyze_member_movement' ---
global_previous_positions_for_activities = {}
global_pose_history_for_member_movement = defaultdict(lambda: [])

# --- Funções Auxiliares de Processamento (Mantidas as suas versões existentes) ---

def analyze_emotion_deepface(face_image):
    if face_image is None or face_image.size == 0:
        return "neutral", 0.0
    try:
        # DeepFace.analyze irá tentar detectar o rosto dentro da face_image fornecida.
        # Se face_image for o person_crop_rgb (caixa YOLO), DeepFace vai procurar o rosto lá dentro.
        results = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent=True)

        if results and isinstance(results, list) and len(results) > 0:
            dominant_emotion = results[0]['dominant_emotion']
            emotion_score = results[0]['emotion'][dominant_emotion]
            return dominant_emotion, emotion_score
    except Exception as e:
        # print(f"Erro na análise de emoção: {e}") # Descomente para depurar erros
        pass
    return "neutral", 0.0

def get_landmark_coords(landmarks, landmark_enum, image_width, image_height, min_visibility=MIN_LANDMARK_VISIBILITY):
    if not landmarks or landmark_enum.value >= len(landmarks):
        return None
    lm = landmarks[landmark_enum.value]
    if lm.visibility > min_visibility:
        return (lm.x * image_width, lm.y * image_height)
    return None

def are_any_landmarks_visible(landmarks, landmark_enums, min_visibility=MIN_LANDMARK_VISIBILITY):
    if not landmarks:
        return False
    for enum in landmark_enums:
        if enum.value < len(landmarks) and landmarks[enum.value].visibility > min_visibility:
            return True
    return False

def adjust_landmarks_to_full_frame(landmarks, crop_x1, crop_y1, crop_width, crop_height, frame_width, frame_height):
    adjusted_landmarks_list = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        new_lm = landmark_pb2.NormalizedLandmark()
        new_lm.x = (lm.x * crop_width + crop_x1) / frame_width
        new_lm.y = (lm.y * crop_height + crop_y1) / frame_height
        new_lm.z = lm.z
        new_lm.visibility = getattr(lm, 'visibility', 0.0)
        adjusted_landmarks_list.landmark.append(new_lm)
    return adjusted_landmarks_list

def calculate_angle(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return None

    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- Função de Detecção de Posição (mantida) ---

def detect_position(landmarks, image_width, image_height, bbox_width, bbox_height, mp_pose_module):
    if not landmarks:
        return "Desconhecido"

    get_lm_xy = lambda lm_enum: get_landmark_coords(landmarks, lm_enum, image_width, image_height)

    nose = get_lm_xy(mp_pose_module.PoseLandmark.NOSE)
    left_shoulder = get_lm_xy(mp_pose_module.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_lm_xy(mp_pose_module.PoseLandmark.RIGHT_SHOULDER)
    left_hip = get_lm_xy(mp_pose_module.PoseLandmark.LEFT_HIP)
    right_hip = get_lm_xy(mp_pose_module.PoseLandmark.RIGHT_HIP)
    left_knee = get_lm_xy(mp_pose_module.PoseLandmark.LEFT_KNEE)
    right_knee = get_lm_xy(mp_pose_module.PoseLandmark.RIGHT_KNEE)
    left_ankle = get_lm_xy(mp_pose_module.PoseLandmark.LEFT_ANKLE)
    right_ankle = get_lm_xy(mp_pose_module.PoseLandmark.RIGHT_ANKLE)

    avg_shoulder_y = np.mean([lm[1] for lm in [left_shoulder, right_shoulder] if lm]) if any([left_shoulder, right_shoulder]) else None
    avg_hip_y = np.mean([lm[1] for lm in [left_hip, right_hip] if lm]) if any([left_hip, right_hip]) else None
    avg_knee_y = np.mean([lm[1] for lm in [left_knee, right_knee] if lm]) if any([left_knee, right_knee]) else None
    avg_ankle_y = np.mean([lm[1] for lm in [left_ankle, right_ankle] if lm]) if any([left_ankle, right_ankle]) else None

    lower_body_landmarks_check = [
        mp_pose_module.PoseLandmark.LEFT_HIP, mp_pose_module.PoseLandmark.RIGHT_HIP,
        mp_pose_module.PoseLandmark.LEFT_KNEE, mp_pose_module.PoseLandmark.RIGHT_KNEE,
        mp_pose_module.PoseLandmark.LEFT_ANKLE, mp_pose_module.PoseLandmark.RIGHT_ANKLE,
        mp_pose_module.PoseLandmark.LEFT_HEEL, mp_pose_module.PoseLandmark.RIGHT_HEEL,
        mp_pose_module.PoseLandmark.LEFT_FOOT_INDEX, mp_pose_module.PoseLandmark.RIGHT_FOOT_INDEX
    ]
    is_any_lower_body_lm_visible = are_any_landmarks_visible(landmarks, lower_body_landmarks_check)

    aspect_ratio_bbox = bbox_width / bbox_height if bbox_height > 0 else 0

    pose_vertical_height = bbox_height
    if avg_hip_y is not None and avg_ankle_y is not None:
        pose_vertical_height = abs(avg_hip_y - avg_ankle_y) * 2
    elif avg_shoulder_y is not None and avg_hip_y is not None:
        pose_vertical_height = abs(avg_shoulder_y - avg_hip_y) * 2

    if aspect_ratio_bbox > THRESHOLD_LYING_ASPECT_RATIO and pose_vertical_height < image_height * THRESHOLD_LYING_VERTICAL_HEIGHT_RATIO:
        return "Deitado"

    if avg_hip_y is not None and avg_knee_y is not None:
        if avg_knee_y > avg_hip_y:
            if avg_ankle_y is not None and avg_ankle_y > avg_knee_y:
                hip_ankle_dist_y = abs(avg_ankle_y - avg_hip_y)
                if bbox_height > 0 and (hip_ankle_dist_y / bbox_height) > STANDING_LEG_TO_BBOX_RATIO_MIN:
                    return "Em Pé"
            else:
                hip_knee_dist_y = abs(avg_knee_y - avg_hip_y)
                if bbox_height > 0 and (hip_knee_dist_y / bbox_height) > STANDING_HIP_KNEE_TO_BBOX_RATIO_MIN:
                    if (avg_shoulder_y is not None and avg_shoulder_y < avg_hip_y) or \
                       (bbox_height / image_height) > 0.4:
                        return "Em Pé"

    is_face_visible = (nose is not None)
    main_upper_body_non_hand_landmarks = [
        mp_pose_module.PoseLandmark.LEFT_SHOULDER, mp_pose_module.PoseLandmark.RIGHT_SHOULDER,
        mp_pose_module.PoseLandmark.LEFT_HIP, mp_pose_module.PoseLandmark.RIGHT_HIP
    ]
    is_any_main_upper_body_non_hand_lm_visible = are_any_landmarks_visible(landmarks, main_upper_body_non_hand_landmarks)

    if is_face_visible or is_any_main_upper_body_non_hand_lm_visible:
        if not is_any_lower_body_lm_visible:
            if bbox_height > image_height * SITTING_BBOX_HEIGHT_MIN_RATIO and bbox_height / image_height < 0.6:
                return "Sentado"
        elif avg_knee_y is not None and avg_hip_y is not None:
            knee_hip_diff = abs(avg_knee_y - avg_hip_y)
            if bbox_height > 0 and (knee_hip_diff / bbox_height) < SITTING_KNEE_HIP_MAX_DIST_RATIO:
                return "Sentado"

            if avg_knee_y < avg_hip_y:
                return "Sentado"

            ang_joelho_esq = calculate_angle(left_hip, left_knee, left_ankle)
            ang_joelho_dir = calculate_angle(right_hip, right_knee, right_ankle)
            ang_joelho = None
            if ang_joelho_esq is not None and ang_joelho_dir is not None:
                ang_joelho = min(ang_joelho_esq, ang_joelho_dir)
            elif ang_joelho_esq is not None:
                ang_joelho = ang_joelho_esq
            elif ang_joelho_dir is not None:
                ang_joelho = ang_joelho_dir

            if ang_joelho is not None and ANGLE_KNEE_SITTING_MIN <= ang_joelho <= ANGLE_KNEE_SITTING_MAX:
                return "Sentado"

            ang_quadril_esq = None
            ang_quadril_dir = None
            if left_hip and left_knee:
                if left_shoulder:
                    ang_quadril_esq = calculate_angle(left_shoulder, left_hip, left_knee)
                elif right_shoulder:
                    ang_quadril_esq = calculate_angle(right_shoulder, left_hip, left_knee)

            if right_hip and right_knee:
                if right_shoulder:
                    ang_quadril_dir = calculate_angle(right_shoulder, right_hip, right_knee)
                elif left_shoulder:
                    ang_quadril_dir = calculate_angle(left_shoulder, right_hip, right_knee)

            ang_quadril = None
            if ang_quadril_esq is not None and ang_quadril_dir is not None:
                ang_quadril = min(ang_quadril_esq, ang_quadril_dir)
            elif ang_quadril_esq is not None:
                ang_quadril = ang_quadril_esq
            elif ang_quadril_dir is not None:
                ang_quadril = ang_quadril_dir

            if ang_quadril is not None and ANGLE_HIP_SITTING_MIN <= ang_quadril <= ANGLE_HIP_SITTING_MAX:
                return "Sentado"

        elif avg_ankle_y is not None and avg_hip_y is not None:
            ankle_hip_diff = avg_ankle_y - avg_hip_y
            if bbox_height > 0 and (ankle_hip_diff / bbox_height) < SITTING_ANKLE_HIP_MAX_DIST_RATIO:
                return "Sentado"

    return "Desconhecido"


# --- Função: detect_activities ---
def detect_activities(pose_results_person, person_id, mp_pose_module, image_width, image_height):
    activities = []
    global global_previous_positions_for_activities

    if person_id not in global_previous_positions_for_activities:
        global_previous_positions_for_activities[person_id] = {}

    previous_positions_for_this_person = global_previous_positions_for_activities[person_id]

    if pose_results_person and pose_results_person.pose_landmarks:
        landmarks = pose_results_person.pose_landmarks.landmark

        get_lm_xy_vis = lambda lm_enum: get_landmark_coords(landmarks, lm_enum, image_width, image_height, MIN_LANDMARK_VISIBILITY)

        left_hand = get_lm_xy_vis(mp_pose_module.PoseLandmark.LEFT_WRIST)
        right_hand = get_lm_xy_vis(mp_pose_module.PoseLandmark.RIGHT_WRIST)
        left_shoulder = get_lm_xy_vis(mp_pose_module.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_lm_xy_vis(mp_pose_module.PoseLandmark.RIGHT_SHOULDER)
        nose = get_lm_xy_vis(mp_pose_module.PoseLandmark.NOSE)
        left_elbow = get_lm_xy_vis(mp_pose_module.PoseLandmark.LEFT_ELBOW)
        right_elbow = get_lm_xy_vis(mp_pose_module.PoseLandmark.RIGHT_ELBOW)


        # Mão levantada (sem movimento lateral)
        if left_hand and left_shoulder and left_hand[1] < left_shoulder[1] and abs(left_hand[0] - left_shoulder[0]) < (0.2 * image_width):
            activities.append("Mao esquerda levantada")
        if right_hand and right_shoulder and right_hand[1] < right_shoulder[1] and abs(right_hand[0] - right_shoulder[0]) < (0.2 * image_width):
            activities.append("Mao direita levantada")

        # Acenando (movimento lateral)
        if left_hand and left_shoulder and left_elbow and \
           abs(left_hand[1] - left_shoulder[1]) < (0.2 * image_height) and \
           abs(left_hand[0] - left_elbow[0]) > (0.3 * image_width):
            activities.append("Acenando com a mao esquerda")
        if right_hand and right_shoulder and right_elbow and \
           abs(right_hand[1] - right_shoulder[1]) < (0.2 * image_height) and \
           abs(right_hand[0] - right_elbow[0]) > (0.3 * image_width):
            activities.append("Acenando com a mao direita")

        # Aperto de mão (mãos estendidas uma em direção à outra)
        if left_hand and right_hand and \
           abs(left_hand[0] - right_hand[0]) < (0.2 * image_width) and \
           abs(left_hand[1] - right_hand[1]) < (0.2 * image_height):

            left_hand_prev = previous_positions_for_this_person.get("left_hand", (None, None))
            right_hand_prev = previous_positions_for_this_person.get("right_hand", (None, None))

            left_move = 0
            if left_hand_prev[0] is not None and left_hand_prev[1] is not None:
                left_move = np.linalg.norm(np.array(left_hand) - np.array(left_hand_prev))

            right_move = 0
            if right_hand_prev[0] is not None and right_hand_prev[1] is not None:
                right_move = np.linalg.norm(np.array(right_hand) - np.array(right_hand_prev))

            if left_move > (0.01 * image_width) or right_move > (0.01 * image_width):
                activities.append("Aperto de mao")

            previous_positions_for_this_person["left_hand"] = left_hand
            previous_positions_for_this_person["right_hand"] = right_hand

        # Dançando, vários movimentos consideráveis na cena.
        current_keypoints_for_dancing = {
            mp_pose_module.PoseLandmark.NOSE: nose,
            mp_pose_module.PoseLandmark.LEFT_SHOULDER: left_shoulder,
            mp_pose_module.PoseLandmark.RIGHT_SHOULDER: right_shoulder,
            mp_pose_module.PoseLandmark.LEFT_WRIST: left_hand,
            mp_pose_module.PoseLandmark.RIGHT_WRIST: right_hand
        }

        movement_threshold_dancing_pixels = 0.3 * image_width
        total_movement = 0

        for kp_enum, current_pos in current_keypoints_for_dancing.items():
            if current_pos:
                prev_pos = previous_positions_for_this_person.get(kp_enum.name, None)
                if prev_pos:
                    total_movement += np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
                previous_positions_for_this_person[kp_enum.name] = current_pos

        if total_movement > movement_threshold_dancing_pixels:
            activities.append("Dancando")

    if not activities:
        activities.append("Atividade nao detectada")

    return activities

# --- Função: analyze_member_movement ---
def analyze_member_movement(current_pose_landmarks, prev_pose_landmarks, frame_width, frame_height, movement_threshold=MOVEMENT_THRESHOLD_MEMBERS):
    moving_members = []

    if not current_pose_landmarks or not prev_pose_landmarks:
        return moving_members

    landmarks_curr = current_pose_landmarks.landmark
    landmarks_prev = prev_pose_landmarks.landmark

    member_landmarks = {
        "Cabeca": [mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_EYE, mp.solutions.pose.PoseLandmark.RIGHT_EYE,
                   mp.solutions.pose.PoseLandmark.LEFT_EAR, mp.solutions.pose.PoseLandmark.RIGHT_EAR,
                   mp.solutions.pose.PoseLandmark.MOUTH_LEFT, mp.solutions.pose.PoseLandmark.MOUTH_RIGHT],
        "Tronco": [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                   mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP],
        "Braco Esquerdo": [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                           mp.solutions.pose.PoseLandmark.LEFT_WRIST],
        "Braco Direito": [mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                          mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
        "Perna Esquerda": [mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE,
                           mp.solutions.pose.PoseLandmark.LEFT_ANKLE],
        "Perna Direita": [mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
                          mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
    }

    visibility_threshold = 0.6

    for member, kps in member_landmarks.items():
        total_displacement_for_member = 0
        num_visible_kps_for_member = 0

        for kp in kps:
            if kp.value < len(landmarks_curr) and kp.value < len(landmarks_prev):
                if landmarks_curr[kp].visibility > visibility_threshold and \
                   landmarks_prev[kp].visibility > visibility_threshold:

                    current_x = landmarks_curr[kp].x * frame_width
                    current_y = landmarks_curr[kp].y * frame_height
                    prev_x = landmarks_prev[kp].x * frame_width
                    prev_y = landmarks_prev[kp].y * frame_height

                    distance = np.linalg.norm(np.array([current_x, current_y]) - np.array([prev_x, prev_y]))
                    total_displacement_for_member += distance
                    num_visible_kps_for_member += 1

        if num_visible_kps_for_member > 0:
            average_displacement = total_displacement_for_member / num_visible_kps_for_member
            if average_displacement > movement_threshold:
                moving_members.append(member)

    return moving_members


# --- Função Principal de Processamento de Vídeo ---

def process_video_with_yolo_and_mediapipe_and_deepface(input_video_path, output_video_path):
    if not os.path.exists(input_video_path):
        print(f"Erro: O arquivo de vídeo de entrada não foi encontrado em '{input_video_path}'")
        return

    yolo_model = YOLO('yolov8m.pt')
    mp_pose_instance = mp.solutions.pose
    pose_estimator = mp_pose_instance.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_hands_instance = mp.solutions.hands
    hands_detector = mp_hands_instance.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: '{input_video_path}'")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Iniciando o processamento do vídeo '{input_video_path}'...")
    print(f"Salvando o vídeo processado em '{output_video_path}'")

    frame_count = 0
    overall_emotion_counts = defaultdict(int)
    overall_position_counts = defaultdict(int)
    overall_activity_counts = defaultdict(int)
    overall_member_movement_counts = defaultdict(int)

    global global_previous_positions_for_activities
    global global_pose_history_for_member_movement
    global_previous_positions_for_activities = {}
    global_pose_history_for_member_movement = defaultdict(lambda: [])

    # --- NOVO: Define a pasta de saída para as imagens e a cria se não existir ---
    # Obtém o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Cria o caminho completo para a subpasta "imagens"
    output_image_dir = os.path.join(script_dir, "imagens")
    # Cria a pasta se ela não existir
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Imagens de depuração serão salvas em: '{output_image_dir}'")
    # --- FIM DO NOVO ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Remova ou descomente conforme a orientação real do seu vídeo!
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        current_frame_height, current_frame_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_yolo = yolo_model.track(frame_rgb, persist=True, classes=[0], verbose=False, conf=0.5, iou=0.7)

        if results_yolo[0].boxes.id is not None:
            track_ids = results_yolo[0].boxes.id.int().cpu().tolist()
            boxes = results_yolo[0].boxes.xyxy.int().cpu().tolist()
            confs = results_yolo[0].boxes.conf.float().cpu().tolist()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                person_id = track_ids[i]
                confidence = confs[i]

                bbox_width = x2 - x1
                bbox_height = y2 - y1

                crop_x1 = max(0, x1)
                crop_y1 = max(0, y1)
                crop_x2 = min(current_frame_width, x2)
                crop_y2 = min(current_frame_height, y2)

                current_crop_width = crop_x2 - crop_x1
                current_crop_height = crop_y2 - crop_y1

                person_crop_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

                # --- LINHA ADICIONADA/MODIFICADA PARA SALVAR IMAGENS ---
                # Apenas salve se o recorte for válido
                if person_crop_rgb.size > 0:
                    # Converte de volta para BGR antes de salvar, pois o OpenCV espera BGR para salvar
                    cv2.imwrite(os.path.join(output_image_dir, f"{frame_count:04d}_person_{person_id}.jpg"),
                                cv2.cvtColor(person_crop_rgb, cv2.COLOR_RGB2BGR))
                # --- FIM DA LINHA ADICIONADA/MODIFICADA --

                if current_crop_width > 0 and current_crop_height > 0:
                    results_mediapipe_person = pose_estimator.process(person_crop_rgb)
                    results_mediapipe_hands = hands_detector.process(person_crop_rgb)

                    adjusted_landmarks_for_logic = None
                    if results_mediapipe_person and results_mediapipe_person.pose_landmarks:
                        adjusted_landmarks_for_logic = adjust_landmarks_to_full_frame(
                            results_mediapipe_person.pose_landmarks.landmark,
                            crop_x1, crop_y1, current_crop_width, current_crop_height,
                            current_frame_width, current_frame_height
                        )
                        global_pose_history_for_member_movement[person_id].append(adjusted_landmarks_for_logic)
                        if len(global_pose_history_for_member_movement[person_id]) > fps:
                            global_pose_history_for_member_movement[person_id].pop(0)

                    person_position = "Desconhecido"
                    if adjusted_landmarks_for_logic:
                        person_position = detect_position(
                            adjusted_landmarks_for_logic.landmark,
                            current_frame_width,
                            current_frame_height,
                            bbox_width,
                            bbox_height,
                            mp_pose_instance
                        )
                    overall_position_counts[person_position] += 1

                    detected_activities_this_person = []
                    if results_mediapipe_person:
                        detected_activities_this_person = detect_activities(
                            results_mediapipe_person,
                            person_id,
                            mp_pose_instance,
                            current_frame_width,
                            current_frame_height
                        )
                        for activity in detected_activities_this_person:
                            overall_activity_counts[activity] += 1

                    moving_members_this_person = []
                    if person_id in global_pose_history_for_member_movement and \
                       len(global_pose_history_for_member_movement[person_id]) >= 2:

                        current_person_pose_lms = global_pose_history_for_member_movement[person_id][-1]
                        prev_person_pose_lms = global_pose_history_for_member_movement[person_id][-2]

                        moving_members_this_person = analyze_member_movement(
                            current_person_pose_lms,
                            prev_person_pose_lms,
                            current_frame_width,
                            current_frame_height
                        )
                        for member in moving_members_this_person:
                            overall_member_movement_counts[member] += 1


                    dominant_emotion, emotion_score = analyze_emotion_deepface(person_crop_rgb)
                    overall_emotion_counts[dominant_emotion] += 1

                    # --- DESENHAR BOUNDING BOX E RÓTULOS DENTRO DA BOX ---
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

                    labels_to_draw = []
                    labels_to_draw.append(f'ID: {person_id} Conf: {confidence:.2f}')
                    labels_to_draw.append(f'Posicao: {person_position}')
                    labels_to_draw.append(f'Emo: {dominant_emotion} ({emotion_score:.1f}%)')

                    # Removida a condição extra, "Atividade nao detectada" agora será exibido se for o caso
                    if detected_activities_this_person:
                        labels_to_draw.append(f'Atividade: {", ".join(detected_activities_this_person)}')
                    if moving_members_this_person:
                        labels_to_draw.append(f'Movimento: {", ".join(moving_members_this_person)}')

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 1
                    padding_x = 5
                    padding_y = 5
                    line_height_increment = 20

                    # Posição Y inicial para o primeiro rótulo dentro da box
                    current_text_y = y1 + padding_y + (text_height if 'text_height' in locals() else line_height_increment) # Ajuste inicial

                    for idx, label_text in enumerate(labels_to_draw):
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

                        # Definir a cor do texto
                        color = (255, 255, 255) # Branco padrão
                        if "Emo:" in label_text:
                            color = (0, 255, 255) # Ciano
                        elif "Posicao:" in label_text:
                            color = (0, 200, 0) # Verde
                        elif "Atividade:" in label_text:
                            color = (255, 0, 255) # Magenta
                        elif "Movimento:" in label_text:
                            color = (255, 150, 0) # Laranja

                        # Calcular as coordenadas do retângulo de fundo
                        bg_x1 = x1 + padding_x - 2
                        bg_y1 = current_text_y - text_height
                        bg_x2 = x1 + padding_x + text_width + 2
                        bg_y2 = current_text_y + baseline

                        # Verificar se o retângulo de fundo e o texto cabem dentro da bounding box
                        # Aumente a margem de erro ou torne a condição mais suave, se necessário
                        if bg_x2 < x2 and bg_y2 < y2 + 5: # Permite um pequeno "vazamento" na parte inferior para evitar corte prematuro
                            # Criar uma cópia do ROI para desenhar o fundo semi-transparente
                            # Isso é mais robusto do que o método alpha direto
                            sub_img = frame[bg_y1:bg_y2, bg_x1:bg_x2]
                            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0 # Cor preta

                            res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.4, 1.0) # Ajuste 0.6 e 0.4 para mais/menos transparência
                            frame[bg_y1:bg_y2, bg_x1:bg_x2] = res

                            cv2.putText(frame, label_text, (x1 + padding_x, current_text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
                            current_text_y += line_height_increment
                        else:
                            # Se não couber, podemos parar de desenhar para evitar bagunça
                            break

                    # Desenhar Landmarks do MediaPipe Pose
                    if adjusted_landmarks_for_logic:
                        mp_drawing.draw_landmarks(
                            frame,
                            adjusted_landmarks_for_logic,
                            mp_pose_instance.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )

                    # Desenhar Landmarks das Mãos
                    if results_mediapipe_hands and results_mediapipe_hands.multi_hand_landmarks:
                        for hand_landmarks in results_mediapipe_hands.multi_hand_landmarks:
                            adjusted_hand_landmarks = adjust_landmarks_to_full_frame(
                                hand_landmarks.landmark,
                                crop_x1, crop_y1, current_crop_width, current_crop_height,
                                current_frame_width, current_frame_height
                            )
                            mp_drawing.draw_landmarks(
                                frame,
                                adjusted_hand_landmarks,
                                mp_hands_instance.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )

        # Exibição e Salvamento
        cv2.imshow('Video Processado - Pressione Q para Sair', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processamento interrompido pelo usuário.")
            break

    # Liberação de Recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose_estimator.close()
    hands_detector.close()

    print(f"Processamento concluído. '{frame_count}' quadros processados.")
    print(f"Vídeo de saída salvo como '{output_video_path}'")

    # --- Gerar Relatório Final em arquivo de texto ---
    # Gerar o nome do arquivo com a data e hora atuais
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"relatorio_movimentos_{current_datetime}.txt"

    with open(report_filename, "w", encoding="utf-8") as report_file:
        report_file.write("--- Relatório de Análise ---\n")
        report_file.write(f"Total de quadros processados: {frame_count}\n\n")

        report_file.write("Resumo das Emoções Detectadas:\n")
        total_emotions = sum(overall_emotion_counts.values())
        for emotion, count in sorted(overall_emotion_counts.items()):
            percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
            report_file.write(f"- {emotion.capitalize()}: {count} frames ({percentage:.2f}%)\n")

        report_file.write("\nResumo das Posições (Posturas) Detectadas:\n")
        total_positions = sum(overall_position_counts.values())
        for position, count in sorted(overall_position_counts.items()):
            percentage = (count / total_positions) * 100 if total_positions > 0 else 0
            report_file.write(f"- {position}: {count} frames ({percentage:.2f}%)\n")

        report_file.write("\nResumo das Atividades Específicas Detectadas:\n")
        total_activities = sum(overall_activity_counts.values())
        for activity, count in sorted(overall_activity_counts.items()):
            percentage = (count / total_activities) * 100 if total_activities > 0 else 0
            report_file.write(f"- {activity}: {count} frames ({percentage:.2f}%)\n")

        report_file.write("\nResumo do Movimento de Membros:\n")
        total_member_movements = sum(overall_member_movement_counts.values())
        for member, count in sorted(overall_member_movement_counts.items()):
            percentage = (count / total_member_movements) * 100 if total_member_movements > 0 else 0
            report_file.write(f"- {member}: {count} frames ({percentage:.2f}%)\n")

    print(f"\nRelatório de análise salvo em: '{report_filename}'")


# --- Executar o script ---
if __name__ == "__main__":
    input_video = 'tc_video.mp4'
    output_video = 'tc_video_output.mp4'
    process_video_with_yolo_and_mediapipe_and_deepface(input_video, output_video)