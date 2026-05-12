import cv2
import numpy as np
import socket
import json
import time
from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES DA ARQUITETURA
# ==========================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
WARP_SIZE = 400
CELL_SIZE = 40 # 400px / 10 células

# Dicionário ArUco para Calibração (Apenas os 4 cantos)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

# Classes do teu modelo YOLOv8
CLASSES = {
    0: "casa",
    1: "arvore",
    2: "personagem",
}

# Tipos dinâmicos (que usam ByteTrack e orientações)
DYNAMIC_TYPES = ["personagem"]

# Configuração UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
seq_num = 0

# Carregar o teu modelo treinado
# Substitui pelo caminho do teu best.pt
model = YOLO("models/best.pt") 

# Memória de posições para calcular orientação ("N", "S", "E", "W")
last_positions = {}

def send_to_unity(status, objs):
    global seq_num
    payload = {
        "seq": seq_num,
        "status": status,
        "objs": objs
    }
    sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
    seq_num += 1

def order_corners(corners, ids):
    """Ordena os 4 ArUco: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    ordered = {}
    for i, id_ in enumerate(ids.flatten()):
        ordered[id_] = corners[i][0].mean(axis=0)
    
    if not all(k in ordered for k in [0, 1, 2, 3]):
        return None
    
    return np.array([ordered[0], ordered[1], ordered[2], ordered[3]], np.float32)

def calculate_orientation(track_id, col, row):
    """Calcula para onde a peça está virada com base na posição anterior"""
    if track_id not in last_positions:
        last_positions[track_id] = (col, row, "N") # Predefinição
        return "N"
    
    last_col, last_row, last_ori = last_positions[track_id]
    
    if col > last_col: ori = "E"
    elif col < last_col: ori = "W"
    elif row > last_row: ori = "S"
    elif row < last_row: ori = "N"
    else: ori = last_ori # Não se moveu, mantém a orientação
    
    last_positions[track_id] = (col, row, ori)
    return ori

# ==========================================
# LOOP PRINCIPAL DE VISÃO
# ==========================================
cap = cv2.VideoCapture(1) # Muda para o índice da tua câmara externa (0, 1, ou 2)
dst_pts = np.array([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]], np.float32)

print("A aguardar calibração do tabuleiro (4 ArUcos)...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. DETETAR ARUCO PARA HOMOGRAFIA
    corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT)
    
    if ids is None or len(ids) < 4:
        # ArUcos ocluídos ou perdidos - envia estado de oclusão
        send_to_unity("occluded", [])
        cv2.imshow("Visao", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # Calcular Matriz de Homografia
    src_pts = order_corners(corners, ids)
    if src_pts is None:
        continue
        
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_frame = cv2.warpPerspective(frame, matrix, (WARP_SIZE, WARP_SIZE))

    # 2. INFERÊNCIA YOLOv8 + BYTETRACK
    # A configuração tracker="bytetrack.yaml" ativa o tracking nativo
    results = model.track(warped_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    board_objects = []
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
            x1, y1, x2, y2 = box
            
            # 3. CÁLCULO DE POSIÇÃO (Centro-Base)
            cx = (x1 + x2) / 2.0
            cy = y2 # Os pés da peça
            
            col = int(cx // CELL_SIZE)
            row = int(cy // CELL_SIZE)
            
            # Garantir que não excede a grelha devido a ruído
            col = max(0, min(9, col))
            row = max(0, min(9, row))
            
            type_name = CLASSES.get(cls_id, "desconhecido")
            
            # 4. GESTÃO DE IDs E ORIENTAÇÃO
            if type_name in DYNAMIC_TYPES:
                # Dinâmicos: ID vem do ByteTrack
                obj_id = track_id
                orientation = calculate_orientation(track_id, col, row)
            else:
                # Estáticos: ID é baseado na coordenada (col*10 + row)
                obj_id = col * 10 + row
                orientation = None
            
            board_objects.append({
                "id": int(obj_id),
                "t": type_name,
                "p": [int(col), int(row)],
                "o": orientation
            })
            
            # Desenhar para debug visual
            cv2.rectangle(warped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(warped_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(warped_frame, f"{type_name} {col},{row}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 5. ENVIAR DADOS
    # Envia o primeiro pacote como reset, ou o estado normal ("ok")
    status_msg = "reset" if seq_num == 0 else "ok"
    send_to_unity(status_msg, board_objects)

    # Visualização de Debug
    cv2.imshow("Tabuleiro Retificado", warped_frame)
    
    # Limita o envio para não sobrecarregar o Unity (~10-15 FPS é ideal)
    time.sleep(0.05) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()