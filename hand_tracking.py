"""
Hand Tracking com MediaPipe + OpenCV
Compatível com mediapipe >= 0.9 e >= 0.10 (detecta automaticamente).

Instalação:
    pip install opencv-python mediapipe==0.10.9

Uso:
    python hand_tracking.py

Controles:
    Q  - Sair
    S  - Salvar screenshot
    D  - Alternar modo debug (exibe IDs das landmarks)
    +  - Aumentar espessura das linhas
    -  - Diminuir espessura das linhas
"""

import cv2
import mediapipe as mp
import time
import os
import sys

# ── Detecta versão da API ─────────────────────
mp_version = tuple(int(x) for x in mp.__version__.split(".")[:2])
USE_NEW_API = mp_version >= (0, 10)
print(f"MediaPipe {mp.__version__} → usando {'nova API (>=0.10)' if USE_NEW_API else 'API clássica (<0.10)'}")

# ──────────────────────────────────────────────
#  Cores por dedo (BGR)
# ──────────────────────────────────────────────
FINGER_COLORS = {
    "thumb":  (255, 100,  50),
    "index":  ( 50, 255, 100),
    "middle": ( 50, 100, 255),
    "ring":   (255,  50, 200),
    "pinky":  (  0, 220, 220),
    "palm":   (200, 200, 200),
}

CONNECTIONS = {
    "thumb":  [(0,1),(1,2),(2,3),(3,4)],
    "index":  [(0,5),(5,6),(6,7),(7,8)],
    "middle": [(0,9),(9,10),(10,11),(11,12)],
    "ring":   [(0,13),(13,14),(14,15),(15,16)],
    "pinky":  [(0,17),(17,18),(18,19),(19,20)],
    "palm":   [(5,9),(9,13),(13,17)],
}


def landmarks_to_pts(landmarks, h, w):
    return {idx: (int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(landmarks)}


def draw_hand(frame, pts, thickness=2, debug=False):
    for finger, conns in CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        for a, b in conns:
            cv2.line(frame, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    for idx, (x, y) in pts.items():
        r = thickness + (5 if idx in (4, 8, 12, 16, 20) else 3)
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (80, 80, 80), 1, cv2.LINE_AA)
        if debug:
            cv2.putText(frame, str(idx), (x+6, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,0), 1, cv2.LINE_AA)


def draw_label(frame, pts, label):
    wx, wy = pts[0]
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (wx-4, wy-lh-10), (wx+lw+4, wy-2), (30,30,30), -1)
    cv2.putText(frame, label, (wx, wy-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,180), 1, cv2.LINE_AA)


def draw_hud(frame, fps, num_hands, thickness, debug):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8,8), (265,122), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate([
        f"FPS: {fps:.1f}",
        f"Maos detectadas: {num_hands}",
        f"Espessura: {thickness}  (+/- para ajustar)",
        f"Debug: {'ON' if debug else 'OFF'}  (D para alternar)",
        "Q=sair  S=screenshot",
    ]):
        cv2.putText(frame, text, (14, 30+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════
#  API CLÁSSICA  (mediapipe < 0.10)
# ══════════════════════════════════════════════
def run_old_api(cap, state):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        num_hands = 0

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            for i, hl in enumerate(results.multi_hand_landmarks):
                pts = landmarks_to_pts(hl.landmark, h, w)
                draw_hand(frame, pts, state["thickness"], state["debug"])

                # DETECÇÃO DE GESTO
                if is_hand_open(pts):
                    texto = "MAO ABERTA"

                elif is_hand_closed(pts):
                    texto = "MAO FECHADA"

                elif is_index_up(pts):
                    texto = "INDICADOR"

                else:
                    texto = ""

                # MOSTRAR TEXTO
                if texto:
                    x, y = pts[0]
                    cv2.putText(frame, texto, (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # LABEL DA MÃO (Left/Right)
                if results.multi_handedness:
                    c = results.multi_handedness[i].classification[0]
                    draw_label(frame, pts, f"{c.label} ({c.score:.0%})")

        # FPS E HUD
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        draw_hud(frame, fps, num_hands, state["thickness"], state["debug"])
        cv2.imshow("Hand Tracking", frame)

        if handle_keys(cv2.waitKey(1) & 0xFF, state, frame):
            break

    hands.close()


# ══════════════════════════════════════════════
#  NOVA API  (mediapipe >= 0.10)
# ══════════════════════════════════════════════
def run_new_api(cap, state):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"Baixando modelo MediaPipe Tasks em '{model_path}' ...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Modelo baixado!")
        except Exception as e:
            print(f"[ERRO] Falha ao baixar modelo: {e}")
            print(f"Baixe manualmente: {url}")
            sys.exit(1)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    prev_time = time.time()

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_img)
            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

            if result.hand_landmarks:
                for i, hl in enumerate(result.hand_landmarks):
                    pts = landmarks_to_pts(hl, h, w)
                    draw_hand(frame, pts, state["thickness"], state["debug"])

                    # 👇 AQUI (mesmo nível do draw_hand)
                    if is_hand_open(pts):
                        texto = "MAO ABERTA"

                    elif is_hand_closed(pts):
                        texto = "MAO FECHADA"

                    elif is_index_up(pts):
                        texto = "INDICADOR"

                    else:
                        texto = ""

                    if texto:
                        x, y = pts[0]

                        # fundo escuro
                        cv2.rectangle(frame, (x-5, y-45), (x+200, y-10), (0,0,0), -1)

                        # texto
                        cv2.putText(frame, texto, (x, y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        

                    if result.handedness and i < len(result.handedness):
                        cat = result.handedness[i][0]
                        draw_label(frame, pts, f"{cat.display_name} ({cat.score:.0%})")

                    if result.handedness and i < len(result.handedness):
                        cat = result.handedness[i][0]
                        draw_label(frame, pts, f"{cat.display_name} ({cat.score:.0%})")

            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            draw_hud(frame, fps, num_hands, state["thickness"], state["debug"])
            cv2.imshow("Hand Tracking", frame)

            if handle_keys(cv2.waitKey(1) & 0xFF, state, frame):
                break

# ══════════════════════════════════════════════
#  Teclado
# ══════════════════════════════════════════════
def handle_keys(key, state, frame):
    """Retorna True se deve encerrar."""
    if key == ord("q"):
        print("Encerrando...")
        return True
    elif key == ord("s"):
        os.makedirs("screenshots", exist_ok=True)
        fn = os.path.join("screenshots", f"hand_{int(time.time())}.png")
        cv2.imwrite(fn, frame)
        print(f"Screenshot salvo: {fn}")
    elif key == ord("d"):
        state["debug"] = not state["debug"]
        print(f"Debug: {'ON' if state['debug'] else 'OFF'}")
    elif key in (ord("+"), ord("=")):
        state["thickness"] = min(state["thickness"] + 1, 8)
    elif key == ord("-"):
        state["thickness"] = max(state["thickness"] - 1, 1)
    return False

def is_hand_open(pts):
    fingers = []

    # Indicador
    fingers.append(pts[8][1] < pts[6][1])
    # Médio
    fingers.append(pts[12][1] < pts[10][1])
    # Anelar
    fingers.append(pts[16][1] < pts[14][1])
    # Mindinho
    fingers.append(pts[20][1] < pts[18][1])

    return all(fingers)

def is_hand_closed(pts):
    return (
        pts[8][1] > pts[6][1] and
        pts[12][1] > pts[10][1] and
        pts[16][1] > pts[14][1] and
        pts[20][1] > pts[18][1]
    )


def is_index_up(pts):
    return (
        pts[8][1] < pts[6][1] and
        pts[12][1] > pts[10][1] and
        pts[16][1] > pts[14][1] and
        pts[20][1] > pts[18][1]
    )


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não encontrada.")
        print("Verifique se a câmera está conectada (e habilitada na VM).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = {"debug": False, "thickness": 2}

    print("=" * 50)
    print("  Hand Tracking — MediaPipe + OpenCV")
    print("  Q=sair | S=screenshot | D=debug | +/-=espessura")
    print("=" * 50)

    try:
        if USE_NEW_API:
            run_new_api(cap, state)
        else:
            run_old_api(cap, state)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()