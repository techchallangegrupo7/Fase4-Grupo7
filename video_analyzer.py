import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace # Usando DeepFace para análise de emoções
import os
import argparse
from datetime import datetime
from tqdm import tqdm # Para barra de progresso

# --- Configurações do MediaPipe ---
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Funções de Análise ---

def analyze_emotion_deepface(face_image):
    """
    Analisa a emoção de uma imagem de rosto usando DeepFace.
    Retorna a emoção dominante e a pontuação, ou None se nenhum rosto for detectado.
    """
    if face_image is None or face_image.size == 0:
        return "neutral", 0.0 # Retorna neutro se a imagem for inválida

    try:
        # DeepFace.analyze espera uma imagem com 3 canais (BGR).
        # Se face_image for grayscale, converte para BGR.
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

        # Usar enforce_detection=False para evitar erros se a detecção facial for fraca
        # actions=['emotion'] para focar apenas na emoção
        results = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent=True) # silent=True para não imprimir no console

        if results and isinstance(results, list) and len(results) > 0:
            # DeepFace pode retornar uma lista de resultados para múltiplos rostos.
            # Se você já passou um ROI de um único rosto, o primeiro resultado deve ser o relevante.
            dominant_emotion = results[0]['dominant_emotion']
            emotion_score = results[0]['emotion'][dominant_emotion]
            return dominant_emotion, emotion_score
        
    except Exception as e:
        #print(f"Erro ao analisar emoção com DeepFace: {e}")
        pass # Ignora erros de detecção ou análise para continuar o processamento

    return "neutral", 0.0 # Retorna neutro se nenhum rosto válido for detectado ou erro

def detect_activity(pose_landmarks, frame_width, frame_height, prev_pose_landmarks=None):
    """
    Detecta atividades com base nos pontos de referência da pose.
    Esta é uma LÓGICA DE EXEMPLO e precisa ser APRIMORADA.

    Argumentos:
        pose_landmarks: Objeto mp.solutions.pose.PoseLandmarks.
        frame_width, frame_height: Dimensões do frame para normalização.
        prev_pose_landmarks: Pontos de referência da pose do frame anterior para análise de movimento.

    Retorna:
        string: A atividade detectada (ex: "reading", "conversing", "walking", "sitting", "unknown").
    """
    if not pose_landmarks:
        return "no_person"

    # Converter landmarks para coordenadas de pixel para facilitar o cálculo
    landmarks = pose_landmarks.landmark
    
    # Exemplo de pontos para análise:
    # Coordenadas são normalizadas para [0,1]. Multiplicamos pela dimensão do frame para obter pixels.
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # Normalizar coordenadas para pixels
    nose_coords = np.array([nose.x * frame_width, nose.y * frame_height])
    ls_coords = np.array([left_shoulder.x * frame_width, left_shoulder.y * frame_height])
    rs_coords = np.array([right_shoulder.x * frame_width, right_shoulder.y * frame_height])
    lw_coords = np.array([left_wrist.x * frame_width, left_wrist.y * frame_height])
    rw_coords = np.array([right_wrist.x * frame_width, right_wrist.y * frame_height])
    lh_coords = np.array([left_hip.x * frame_width, left_hip.y * frame_height])
    rh_coords = np.array([right_hip.x * frame_width, right_hip.y * frame_height])

    # --- Lógica de Exemplo para Atividades ---
    
    # Estimativa de posição vertical do corpo
    avg_shoulder_y = (ls_coords[1] + rs_coords[1]) / 2
    avg_hip_y = (lh_coords[1] + rh_coords[1]) / 2

    # Se a pessoa estiver predominantemente na parte inferior do frame (sentada)
    # ou se a distância entre ombros e quadris for pequena verticalmente
    if avg_hip_y > frame_height * 0.6 or (avg_hip_y - avg_shoulder_y) < (frame_height * 0.25):
        # Provável que esteja sentado
        
        if prev_pose_landmarks:
            prev_nose_coords = np.array([prev_pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame_width, 
                                         prev_pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame_height])
            head_movement = np.linalg.norm(nose_coords - prev_nose_coords)
            
            # Checar movimento de braços/mãos para diferenciar conversação de leitura/escrita
            prev_lw_coords = np.array([prev_pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width, 
                                       prev_pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height])
            prev_rw_coords = np.array([prev_pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width, 
                                       prev_pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height])
            
            wrist_movement_l = np.linalg.norm(lw_coords - prev_lw_coords)
            wrist_movement_r = np.linalg.norm(rw_coords - prev_rw_coords)

            if head_movement < 10 and wrist_movement_l < 15 and wrist_movement_r < 15: # Pequeno movimento
                return "reading/writing" 
            else:
                return "conversing" # Movimento de cabeça/braços sugere interação
        else:
            return "sitting" # Se não houver histórico, assuma sentado pela postura

    # Detecção de Caminhada: Movimento significativo dos quadris ou deslocamento horizontal
    if prev_pose_landmarks:
        prev_lh_coords = np.array([prev_pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width, 
                                   prev_pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height])
        
        hip_movement = np.linalg.norm(lh_coords - prev_lh_coords)
        
        if hip_movement > 20: # Limiar de movimento para caminhada
            return "walking"

    # Atividade geral se não se encaixa nas anteriores
    return "general_activity"

def detect_anomaly(current_pose_landmarks, movement_history, frame_width, frame_height, velocity_threshold=30):
    """
    Detecta anomalias com base em movimentos bruscos ou comportamentos atípicos.
    Esta é uma LÓGICA DE EXEMPLO e precisa ser APRIMORADA.

    Argumentos:
        current_pose_landmarks: Pontos de referência da pose do frame atual.
        movement_history: Lista de objetos pose_landmarks dos frames anteriores.
        frame_width, frame_height: Dimensões do frame para normalização.
        velocity_threshold: Limiar de velocidade em pixels/frame para considerar anomalia.

    Retorna:
        bool: True se uma anomalia for detectada, False caso contrário.
    """
    if not current_pose_landmarks or len(movement_history) < 1:
        return False # Não há histórico suficiente para detectar anomalias

    prev_pose_landmarks = movement_history[-1] # Pega o frame anterior imediato

    # Calcular a distância média de movimento de alguns pontos chave
    keypoints_to_monitor = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_ANKLE, # Pés para detecção de movimento brusco
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]

    total_movement = 0
    num_moved_keypoints = 0

    for kp in keypoints_to_monitor:
        if current_pose_landmarks.landmark[kp].visibility > 0.5 and \
           prev_pose_landmarks.landmark[kp].visibility > 0.5: # Apenas se visível
            
            current_x, current_y = current_pose_landmarks.landmark[kp].x * frame_width, current_pose_landmarks.landmark[kp].y * frame_height
            prev_x, prev_y = prev_pose_landmarks.landmark[kp].x * frame_width, prev_pose_landmarks.landmark[kp].y * frame_height

            distance = np.linalg.norm(np.array([current_x, current_y]) - np.array([prev_x, prev_y]))

            if distance > 0:
                total_movement += distance
                num_moved_keypoints += 1
    
    if num_moved_keypoints > 0:
        average_velocity = total_movement / num_moved_keypoints
        
        # Se a velocidade média for muito alta, considere uma anomalia (gesto brusco)
        # O limiar pode precisar de ajuste fino.
        if average_velocity > velocity_threshold:
            return True
    
    # Futuramente: lógica para comportamentos atípicos que não sejam apenas velocidade
    # Ex: Posições corporais que são estaticamente incomuns para o contexto.
    # Isso exigiria análise de padrões e talvez um modelo de ML.

    return False

def process_video(video_path, output_dir="output"):
    """
    Processa o vídeo para reconhecimento facial, análise de emoções e detecção de atividades.
    Gera um vídeo processado e um relatório.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
        return

    # Obter propriedades do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar o gravador de vídeo de saída (opcional, mas bom para visualização)
    output_video_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    anomalies_detected_count = 0
    
    # Dicionários para armazenar dados para o relatório
    emotional_data = {} # {emotion: count_frames}
    activity_data = {}  # {activity: count_frames}

    # Histórico de landmarks de pose para detecção de anomalias/atividades baseadas em movimento
    movement_history = []
    MAX_HISTORY_FRAMES = int(fps * 0.5) # Guardar meio segundo de histórico (ajustável)

    print(f"Iniciando processamento de vídeo: {video_path}")
    print(f"Dimensões: {frame_width}x{frame_height}, FPS: {fps}, Total de Frames: {total_frames}")

    # Use tqdm para uma barra de progresso visual
    with tqdm(total=total_frames, desc="Processando Frames") as pbar:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
             mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                # Converter BGR para RGB para o MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False # Otimização de performance

                # --- 1. Reconhecimento Facial (usando MediaPipe para detecção e DeepFace para análise) ---
                face_results = face_detection.process(image_rgb)
                
                # --- 2. Análise de Expressões Emocionais ---
                if face_results.detections:
                    for id_detection, detection in enumerate(face_results.detections):
                        mp_drawing.draw_detection(frame, detection) # Desenha o bounding box do rosto
                        
                        # Extrai a região do rosto
                        bboxC = detection.location_data.relative_bounding_box
                        x, y, w_face, h_face = int(bboxC.xmin * frame_width), \
                                               int(bboxC.ymin * frame_height), \
                                               int(bboxC.width * frame_width), \
                                               int(bboxC.height * frame_height)
                        
                        # Garante que as coordenadas são válidas
                        x = max(0, x)
                        y = max(0, y)
                        w_face = min(w_face, frame_width - x)
                        h_face = min(h_face, frame_height - y)

                        face_roi = frame[y:y+h_face, x:x+w_face]

                        if face_roi.size > 0: # Verifica se o ROI não está vazio
                            emotion, score = analyze_emotion_deepface(face_roi)
                            
                            if emotion:
                                emotional_data[emotion] = emotional_data.get(emotion, 0) + 1
                                text_emotion = f"{emotion} ({score:.2f})"
                                cv2.putText(frame, text_emotion, (x, y - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        
                image_rgb.flags.writeable = True # Reativa a gravação para o MediaPipe Pose
                
                # --- 3. Detecção de Atividades ---
                pose_results = pose.process(image_rgb)
                current_pose_landmarks = pose_results.pose_landmarks
                
                activity = "no_activity"
                if current_pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        current_pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # Armazenar histórico para análise de movimento
                    movement_history.append(current_pose_landmarks)
                    if len(movement_history) > MAX_HISTORY_FRAMES:
                        movement_history.pop(0) # Remove o frame mais antigo

                    # Realiza a detecção de atividade
                    prev_pose = movement_history[-2] if len(movement_history) > 1 else None
                    activity = detect_activity(current_pose_landmarks, frame_width, frame_height, prev_pose)
                    activity_data[activity] = activity_data.get(activity, 0) + 1
                    cv2.putText(frame, f"Activity: {activity}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # --- 4. Detecção de Anomalias ---
                    if detect_anomaly(current_pose_landmarks, movement_history, frame_width, frame_height):
                        anomalies_detected_count += 1
                        cv2.putText(frame, "ANOMALY DETECTED!", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    activity_data[activity] = activity_data.get(activity, 0) + 1

                # Escreve o frame processado no vídeo de saída
                out.write(frame)

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
            f.write("  Nenhuma expressão emocional detectada ou analisada.\n")
        
        f.write("\nResumo das Atividades Detectadas:\n")
        if activity_data:
            total_activity_frames = sum(activity_data.values())
            for activity, count in sorted(activity_data.items(), key=lambda item: item[1], reverse=True):
                percentage = (count / total_activity_frames) * 100 if total_activity_frames > 0 else 0
                f.write(f"- {activity.capitalize()}: {count} frames ({percentage:.2f}%)\n")
        else:
            f.write("  Nenhuma atividade detectada.\n")
            
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
    
    args = parser.parse_args()

    # Valida se o arquivo de vídeo existe
    if not os.path.exists(args.video_path):
        print(f"Erro: O arquivo de vídeo '{args.video_path}' não foi encontrado.")
        exit()

    print("\nCertifique-se de ter as seguintes bibliotecas instaladas:")
    print("pip install opencv-python mediapipe numpy deepface tensorflow-cpu tqdm")
    print("\nProcessando o vídeo... Isso pode levar um tempo, dependendo do tamanho do vídeo e da sua máquina.")
    
    process_video(args.video_path, args.output_dir)