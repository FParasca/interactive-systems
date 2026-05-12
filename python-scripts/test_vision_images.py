import cv2
import numpy as np
import socket
import json
import time
import glob
from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES DO AMBIENTE DE TESTE
# ==========================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
WARP_SIZE = 400
CELL_SIZE = 40 # 400px / 10 células

# Caminho para a pasta onde tens as tuas fotos de teste
# (Garante que as fotos têm os 4 ArUcos visíveis)
CAMINHO_FOTOS = "Fotos3/*.jpg" 

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

CLASSES = {
    0: "casa",
    1: "arvore",
    2: "personagem",
  
}

DYNAMIC_TYPES = ["personagem"]

# Configuração UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
seq_num = 0

# Carregar o modelo treinado
model = YOLO("models/best.pt") 

# Memória de posições para calcular orientação
last_positions = {}

def send_to_unity(status, objs):
    global seq_num
    payload = {
        "seq": seq_num,
        "status": status,
        "objs": objs
    }
    sock.sendto(json.dumps(payload).encode("utf-8"), (UDP_IP, UDP_PORT))
    print(f"[UDP] Enviado seq {seq_num} | Status: {status} | Objetos: {len(objs)}")
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
    """Calcula para onde a peça está virada"""
    if track_id not in last_positions:
        last_positions[track_id] = (col, row, "N")
        return "N"
    
    last_col, last_row, last_ori = last_positions[track_id]
    
    if col > last_col: ori = "E"
    elif col < last_col: ori = "W"
    elif row > last_row: ori = "S"
    elif row < last_row: ori = "N"
    else: ori = last_ori 
    
    last_positions[track_id] = (col, row, ori)
    return ori

# ==========================================
# LOOP DE TESTE COM FOTOGRAFIAS ESTÁTICAS
# ==========================================
lista_fotos = sorted(glob.glob(CAMINHO_FOTOS))

if len(lista_fotos) == 0:
    print(f"ERRO: Nenhuma fotografia encontrada em '{CAMINHO_FOTOS}'. Verifica o caminho.")
    exit()

dst_pts = np.array([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]], np.float32)

print(f"A iniciar teste com {len(lista_fotos)} fotografias...")
print("Pressiona QUALQUER TECLA para avançar para a próxima foto. Pressiona 'q' para sair.")

for img_path in lista_fotos:
    print(f"\nA processar: {img_path}")
    frame = cv2.imread(img_path)
    if frame is None: 
        print("Falha ao ler a imagem.")
        continue

    # 1. DETETAR ARUCO
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    
    if ids is None or len(ids) < 4:
        print("ArUcos insuficientes. A enviar estado 'occluded'.")
        send_to_unity("occluded", [])
        cv2.imshow("Debug - Falha ArUco", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'): break
        continue

    # 2. HOMOGRAFIA
    src_pts = order_corners(corners, ids)
    if src_pts is None:
        print("Erro ao ordenar os ArUcos.")
        continue
        
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_frame = cv2.warpPerspective(frame, matrix, (WARP_SIZE, WARP_SIZE))

    # 3. YOLO + BYTETRACK
    results = model.track(warped_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    board_objects = []
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
            x1, y1, x2, y2 = box
            
            # Centro-Base
            cx = (x1 + x2) / 2.0
            cy = y2 
            
            col = int(cx // CELL_SIZE)
            row = int(cy // CELL_SIZE)
            
            col = max(0, min(9, col))
            row = max(0, min(9, row))
            
            type_name = CLASSES.get(cls_id, "desconhecido")
            
            if type_name in DYNAMIC_TYPES:
                obj_id = track_id
                orientation = calculate_orientation(track_id, col, row)
            else:
                obj_id = col * 10 + row
                orientation = None
            
            board_objects.append({
                "id": int(obj_id),
                "t": type_name,
                "p": [int(col), int(row)],
                "o": orientation
            })
            
            # Desenhar Debug
            cv2.rectangle(warped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(warped_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(warped_frame, f"{type_name} {col},{row} (ID:{obj_id})", 
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 4. ENVIAR PARA UNITY
    status_msg = "reset" if seq_num == 0 else "ok"
    send_to_unity(status_msg, board_objects)

    # Mostrar resultado
    cv2.imshow("Tabuleiro Retificado (Teste)", warped_frame)
    
    # Pausa e aguarda que carregues numa tecla para ir para a foto seguinte
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print("Teste interrompido pelo utilizador.")
        break

cv2.destroyAllWindows()
sock.close()
print("Teste concluído.")