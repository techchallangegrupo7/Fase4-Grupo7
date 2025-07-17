import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import os
import argparse
from datetime import datetime
from tqdm import tqdm

# --- Configurações do MediaPipe ---
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles 

# --- Funções de Análise (inalteradas) ---
def analyze_emotion_deepface(face_image):
    """
    Analisa a emoção de uma imagem de rosto usando DeepFace.
    Retorna a emoção dominante e a pontuação, ou None se nenhum rosto for detectado.
    """
    if face_image is None or face_image.size == 0:
        return "neutral", 0.0

    try:
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        results = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent=True)
        if results and isinstance(results, list) and len(results) > 0:
            dominant_emotion = results[0]['dominant_emotion']
            emotion_score = results[0]['emotion'][dominant_emotion]
            return dominant_emotion, emotion_score
    except Exception as e:
        pass
    return "neutral", 0.0

def detect_anomaly(current_pose_landmarks, movement_history, frame_width, frame_height, velocity_threshold=30):
    """
    Detecta anomalias com base em movimentos bruscos ou comportamentos atípicos.
    """
    if not current_pose_landmarks or len(movement_history) < 1:
        return False

    prev_pose_landmarks = movement_history[-1]
    keypoints_to_monitor = [
        mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    total_movement = 0
    num_moved_keypoints = 0
    visibility_threshold = 0.7 # Manter um pouco mais alto para anomalias

    for kp in keypoints_to_monitor:
        if current_pose_landmarks.landmark[kp].visibility > visibility_threshold and \
           prev_pose_landmarks.landmark[kp].visibility > visibility_threshold:
            current_x, current_y = current_pose_landmarks.landmark[kp].x * frame_width, current_pose_landmarks.landmark[kp].y * frame_height
            prev_x, prev_y = prev_pose_landmarks.landmark[kp].x * frame_width, prev_pose_landmarks.landmark[kp].y * frame_height
            distance = np.linalg.norm(np.array([current_x, current_y]) - np.array([prev_x, prev_y]))
            if distance > 0:
                total_movement += distance
                num_moved_keypoints += 1
    
    if num_moved_keypoints > 0:
        average_velocity = total_movement / num_moved_keypoints
        if average_velocity > velocity_threshold:
            return True
    return False

# --- process_video (corrigido e aprimorado) ---
def process_video(video_path, output_dir="output", display_video=True):
    """
    Processa o vídeo para reconhecimento facial, análise de emoções e detecção de atividades.
    Gera um vídeo processado e um relatório. Opcionalmente, exibe o vídeo em tempo real.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    anomalies_detected_count = 0
    emotional_data = {}
    activity_data = {} 

    pose_history = []
    hand_history = []
    MAX_HISTORY_FRAMES = int(fps * 0.5)
    if MAX_HISTORY_FRAMES < 2: MAX_HISTORY_FRAMES = 2

    print(f"Iniciando processamento de vídeo: {video_path}")
    print(f"Dimensões: {frame_width}x{frame_height}, FPS: {fps}, Total de Frames: {total_frames}")

    with tqdm(total=total_frames, desc="Processando Frames") as pbar:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
             mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
             mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands: 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False 

                # --- 1. Reconhecimento Facial e 2. Análise de Expressões Emocionais ---
                face_results = face_detection.process(image_rgb)
                
                if face_results.detections:
                    for detection in face_results.detections:
                        mp_drawing.draw_detection(frame, detection)
                        bboxC = detection.location_data.relative_bounding_box
                        x, y, w_face, h_face = int(bboxC.xmin * frame_width), int(bboxC.ymin * frame_height), \
                                               int(bboxC.width * frame_width), int(bboxC.height * frame_height)
                        x, y = max(0, x), max(0, y)
                        w_face, h_face = min(w_face, frame_width - x), min(h_face, frame_height - y)
                        face_roi = frame[y:y+h_face, x:x+w_face]

                        if face_roi.size > 0:
                            emotion, score = analyze_emotion_deepface(face_roi)
                            if emotion:
                                emotional_data[emotion] = emotional_data.get(emotion, 0) + 1
                                text_emotion = f"{emotion} ({score:.2f})"
                                cv2.putText(frame, text_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                
                image_rgb.flags.writeable = True 
                
                # --- Detecção de Mãos (apenas desenha, sem classificar atividade aqui) ---
                hand_results = hands.process(image_rgb)
                current_hand_landmarks_list = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                 mp_drawing_styles.get_default_hand_landmarks_style(),
                                                 mp_drawing_styles.get_default_hand_connections_style())
                        current_hand_landmarks_list.append(hand_landmarks)

                # --- Detecção de Pose ---
                pose_results = pose.process(image_rgb)
                current_pose_landmarks = pose_results.pose_landmarks
                
                if current_pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        current_pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # --- Armazenar históricos (ainda úteis para anomalias se necessário) ---
                pose_history.append(current_pose_landmarks)
                if len(pose_history) > MAX_HISTORY_FRAMES:
                    pose_history.pop(0)

                hand_history.append(current_hand_landmarks_list)
                if len(hand_history) > MAX_HISTORY_FRAMES:
                    hand_history.pop(0)

                # --- Lógica de Classificação de Atividade Concatenada ---
                
                # Define um limiar de visibilidade para os landmarks da pose
                POSE_VISIBILITY_THRESHOLD = 0.5 

                # Inicializa as flags de detecção de partes do corpo
                detected_parts = []

                # Flags de detecção gerais para facilitar a leitura
                face_detected = bool(face_results.detections)
                pose_detected = bool(current_pose_landmarks) # Definido aqui, antes de ser usado
                hands_detected = bool(current_hand_landmarks_list) # Definido aqui, antes de ser usado


                # Rosto
                if face_detected:
                    detected_parts.append("Rosto")
                
                # Mãos
                if hands_detected: 
                    detected_parts.append("Maos")

                # Tronco e Braços (inferido da pose)
                if pose_detected: # Agora 'pose_detected' está garantido de ser definido
                    landmarks = current_pose_landmarks.landmark
                    
                    # Para TRONCO: Verifica ombros E quadris (pelo menos um de cada lado para robustez)
                    # O tronco é considerado visível se houver ombros E quadris.
                    # Ou, se houver ombros E o centro da parte superior do corpo (mid_shoulder ou algo similar)
                    shoulders_conf = max(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility)
                    hips_conf = max(landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility)
                    
                    tronco_visible = False
                    if shoulders_conf > POSE_VISIBILITY_THRESHOLD and hips_conf > POSE_VISIBILITY_THRESHOLD:
                        tronco_visible = True
                    # Adicionalmente, se houver ombros e o ponto do umbigo/cintura (mid_hip)
                    elif shoulders_conf > POSE_VISIBILITY_THRESHOLD and \
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > POSE_VISIBILITY_THRESHOLD and \
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility > POSE_VISIBILITY_THRESHOLD: # Verificando ambos os quadris para tronco mais completo
                        tronco_visible = True


                    if tronco_visible:
                        detected_parts.append("Tronco")
                    
                    # Para BRAÇOS: Verifica cotovelos e pulsos (pelo menos um de cada lado)
                    # Braços são considerados visíveis se houver cotovelos.
                    # Consideramos a presença de ombros para contextualizar os braços
                    elbows_conf = max(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].visibility,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility)
                    
                    # Braços são visíveis se cotovelos > threshold E (ombros > threshold OU tronco já detectado)
                    if elbows_conf > POSE_VISIBILITY_THRESHOLD and \
                       (shoulders_conf > POSE_VISIBILITY_THRESHOLD or tronco_visible):
                        if "Braços" not in detected_parts: 
                            detected_parts.append("BraCos")


                    # Para PERNAS: Verifica joelhos e tornozelos (pelo menos um de cada lado)
                    knees_conf = max(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility)
                    ankles_conf = max(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility)

                    if knees_conf > POSE_VISIBILITY_THRESHOLD and ankles_conf > POSE_VISIBILITY_THRESHOLD:
                        detected_parts.append("Pernas")
                
                # Constrói o rótulo da atividade
                if detected_parts:
                    current_activity_label = " + ".join(detected_parts)
                else:
                    current_activity_label = "nenhuma_detecao"

                activity_data[current_activity_label] = activity_data.get(current_activity_label, 0) + 1
                cv2.putText(frame, f"Atividade: {current_activity_label.replace('_', ' ').title()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 

                # --- 4. Detecção de Anomalias ---
                # A função detect_anomaly continua usando a lógica anterior de pose history
                if detect_anomaly(current_pose_landmarks, pose_history, frame_width, frame_height):
                    anomalies_detected_count += 1
                    cv2.putText(frame, "ANOMALIA DETECTADA!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                
                out.write(frame)

                # --- EXIBIÇÃO DO VÍDEO EM TEMPO REAL ---
                if display_video:
                    cv2.imshow('Processed Video Feed (Press Q to Quit)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    # --- FINALIZAR: Liberar recursos e fechar janelas ---
    cap.release()
    out.release()
    cv2.destroyAllWindows() 

    # --- 5. Geração de Resumo ---
    report_filename = os.path.join(output_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_filename, "w") as f:
        f.write("--- Relatório de Análise de Vídeo ---\n\n")
        f.write(f"Vídeo Analisado: {os.path.basename(video_path)}\n")
        f.write(f"Data da Análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total de Frames Analisados: {frame_count}\n")
        f.write(f"Número de Anomalias Detectadas: {anomalies_detected_count}\n\n")

        f.write("Resumo das Expressões Emocionais:\n")
        if emotional_data:
            total_emotion_frames = sum(emotional_data.values())
            for emotion, count in sorted(emotional_data.items(), key=lambda item: item[1], reverse=True):
                percentage = (count / total_emotion_frames) * 100 if total_emotion_frames > 0 else 0
                f.write(f"- {emotion.capitalize()}: {count} frames ({percentage:.2f}%)\n")
        else:
            f.write("   Nenhuma expressão emocional detectada ou analisada.\n")
        
        f.write("\nResumo das Atividades Detectadas:\n")
        if activity_data:
            total_activity_frames = sum(activity_data.values())
            for activity, count in sorted(activity_data.items(), key=lambda item: item[1], reverse=True):
                percentage = (count / total_activity_frames) * 100 if total_activity_frames > 0 else 0
                f.write(f"- {activity.replace('_', ' ').title()}: {count} frames ({percentage:.2f}%)\n")
        else:
            f.write("   Nenhuma atividade detectada.\n")
            
        f.write("\n--- Fim do Relatório ---")

    print("\n--- Análise Concluída ---")
    print(f"Vídeo processado salvo em: {output_video_path}")
    print(f"Relatório gerado em: {report_filename}")

# --- Execução Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplicação de Análise de Vídeo para Tech Challenge.")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Caminho completo para o arquivo de vídeo (ex: tc_video.mp4).")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Diretório para salvar o vídeo processado e o relatório. Padrão: 'output'.")
    parser.add_argument("--display_video", action="store_true", 
                        help="Define se o vídeo processado deve ser exibido em tempo real. Padrão: False.")
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Erro: O arquivo de vídeo '{args.video_path}' não foi encontrado.")
        exit()

    print("\nCertifique-se de ter as seguintes bibliotecas instaladas:")
    print("pip install opencv-python mediapipe numpy deepface tensorflow-cpu tqdm")
    print("Para detecção de pose e mãos, você precisará do MediaPipe Hands e Pose, que são incluídos no 'mediapipe'.")
    print("\nProcessando o vídeo... Isso pode levar um tempo, dependendo do tamanho do vídeo e da sua máquina.")
    
    process_video(args.video_path, args.output_dir, args.display_video)