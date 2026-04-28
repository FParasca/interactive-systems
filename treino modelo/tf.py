import cv2
import numpy as np
import os

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
WARP_SIZE  = 400
INPUT_DIR  = "fotos_originais"
OUTPUT_DIR = "fotos_retificadas"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def order_corners(corners, ids):
    ordered = {}
    for i, id_ in enumerate(ids.flatten()):
        ordered[id_] = corners[i][0].mean(axis=0)
    if not all(k in ordered for k in [0,1,2,3]):
        return None
    return np.array([ordered[0], ordered[1], ordered[2], ordered[3]], np.float32)

dst = np.array([[0,0],[WARP_SIZE,0],[WARP_SIZE,WARP_SIZE],[0,WARP_SIZE]], np.float32)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(('.jpg','.png','.jpeg')):
        continue
    frame = cv2.imread(os.path.join(INPUT_DIR, fname))
    corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT)
    if ids is None or len(ids) < 4:
        print(f"ArUco nao detetados em {fname} — ignorado")
        continue
    src = order_corners(corners, ids)
    if src is None:
        continue
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), warped)
    print(f"Retificado: {fname}")

print("Concluido.")
