import cv2
import numpy as np
import mediapipe as mp
from time import time

# --- Config ---
WIN_VIDEO = "Cam (FaceMesh)"
WIN_MAP   = "Face Map (normalized)"
MAP_SIZE  = 512  # lienzo de mapeo

mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
mp_mesh    = mp.solutions.face_mesh

def draw_fps(img, fps):
    text = f"FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

def landmarks_to_canvas(landmarks, w, h, size=MAP_SIZE):
    """
    Recibe landmarks normalizados (x,y en [0,1]) y los dibuja en un lienzo cuadrado.
    También centra y escala manteniendo proporción del rostro detectado.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # Puntos normalizados (0..1). Los convertimos a px en el lienzo
    # Opción simple: mapeo directo x,y → canvas
    pts = []
    for lm in landmarks:
        x = int(lm.x * size)
        y = int(lm.y * size)
        pts.append((x, y))

    # Dibuja puntos
    for (x, y) in pts:
        cv2.circle(canvas, (x, y), 1, (255, 255, 255), -1)

    # Opcional: dibuja contorno aproximado (ojos/nariz/boca) con algunos índices típicos
    # (indices de ejemplo de FaceMesh: ojo der 33, izq 263, nariz 1, boca sup 13, boca inf 14)
    key_idx = [33, 263, 1, 13, 14]
    for i in range(len(key_idx) - 1):
        p1 = pts[key_idx[i]]
        p2 = pts[key_idx[i+1]]
        cv2.line(canvas, p1, p2, (0, 255, 0), 1)

    return canvas

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # en Windows ayuda a abrir más rápido
    if not cap.isOpened():
        print("No pude abrir la cámara.")
        return

    cv2.namedWindow(WIN_VIDEO, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MAP,   cv2.WINDOW_NORMAL)

    # Config FaceMesh: refine_landmarks=True para mejor contorno de ojos/labios
    with mp_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        t0 = time()
        frames = 0
        fps = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frames += 1
            if frames % 10 == 0:
                now = time()
                fps = 10.0 / (now - t0)
                t0 = now

            # BGR -> RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            res = face_mesh.process(rgb)

            # Dibujo por defecto
            out = frame.copy()

            if res.multi_face_landmarks:
                face_landmarks = res.multi_face_landmarks[0]

                # Dibuja la malla en la imagen
                mp_drawing.draw_landmarks(
                    image=out,
                    landmark_list=face_landmarks,
                    connections=mp_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=out,
                    landmark_list=face_landmarks,
                    connections=mp_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                )

                # Genera el "mapa" normalizado en lienzo
                canvas = landmarks_to_canvas(face_landmarks.landmark, w, h, size=MAP_SIZE)

                draw_fps(out, fps)
                draw_fps(canvas, fps)

                cv2.imshow(WIN_VIDEO, out)
                cv2.imshow(WIN_MAP, canvas)
            else:
                draw_fps(out, fps)
                cv2.imshow(WIN_VIDEO, out)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
