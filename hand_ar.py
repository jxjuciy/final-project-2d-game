"""
FACE MARIO - a 2D platformer controlled with your face and hands.

CONTROLS (webcam)
  * TILT HEAD to the LEFT  (ear toward left shoulder)  -> walk left
  * TILT HEAD to the RIGHT (ear toward right shoulder) -> walk right
  * Head straight                                       -> stand still
    (tilt more = walk faster)
  * Close your hand into a FIST           -> PUNCH
  * OPEN YOUR MOUTH                       -> JUMP (keep it open longer = jump higher)

KEYBOARD FALLBACK
  Arrows / A-D = move, SPACE / W = jump, F / J = punch
  C = recalibrate mouth, R = restart, P = pause, ESC = quit

FILES (same folder as this script)
  background.png        your background picture (optional, fallback sky is used)
  face_landmarker.task  downloaded automatically
  hand_landmarker.task  downloaded automatically

POWERS  (hit a gold ? box from below to get one at random)
  Fire Flower     punch throws a fireball
  Wizard Power    a magic shot that clears every enemy in front of you
  Shield          absorbs one hit
  Super Jump      you jump much higher for a while
  Star            short burst where touching enemies defeats them

FILES (same folder as this script)
  background.png        your background picture (optional)
  sprites/enemy_*.png   your enemy pictures (optional, drawn fallback used)

INSTALL
  pip install opencv-python mediapipe pygame numpy
"""

import math
import random
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FACE_MODEL = BASE_DIR / "face_landmarker.task"
HAND_MODEL = BASE_DIR / "hand_landmarker.task"
BACKGROUND_FILE = BASE_DIR / "background.png"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

CAMERA_INDEX = 0
CAM_W, CAM_H = 640, 480

TILE = 48
ROWS = 12
SCREEN_W, SCREEN_H = 1024, TILE * ROWS  # 1024 x 576
FPS = 60

GRAVITY = 2200.0
MAX_FALL = 1100.0
WALK_SPEED = 300.0
ACCEL = 2400.0
FRICTION = 2600.0
JUMP_VEL = 820.0
JUMP_CUT = 0.45          # when mouth closes early, velocity is multiplied by this
COYOTE_TIME = 0.10
JUMP_BUFFER = 0.15
PUNCH_TIME = 0.22
PUNCH_COOLDOWN = 0.30
PUNCH_RANGE = 46
INVINCIBLE_TIME = 1.6

# --- powers ---
POWER_DURATION = {"star": 9.0, "superjump": 12.0, "wizard": 18.0, "fire": 22.0}
POWER_NAMES = {"fire": "FIRE FLOWER", "wizard": "WIZARD POWER",
               "shield": "SHIELD", "superjump": "SUPER JUMP", "star": "STAR"}
FIREBALL_SPEED = 520.0
WIZARD_SPEED = 620.0
SUPER_JUMP_VEL = 1180.0
SPRITE_DIR = BASE_DIR / "sprites"

TILT_DEADZONE = 8.0      # degrees of head tilt ignored around straight
TILT_FULL = 22.0         # degrees of tilt for full walking speed
TILT_RELEASE = 5.0       # hysteresis: stop walking once tilt drops below this
MOUTH_MIN_THRESHOLD = 0.055
MOUTH_EXTRA = 0.045      # threshold = calibrated closed mouth + this

FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
HEAD_SIZE = 56


def ensure_model(path, url):
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {path.name} ...")
    tmp = path.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    print("  done.")


# ------------------------------------------------------------
# VISION THREAD  (camera + face + hands, never blocks the game)
# ------------------------------------------------------------
class VisionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera_ok = False
        self.face_found = False
        self.hand_found = False
        self.mouth_ratio = 0.0
        self.head_roll = 0.0
        self.fist = False
        self.head_rgba = None      # (bytes, w, h)
        self.preview = None        # (bytes, w, h) RGB
        self.fps = 0.0


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


class VisionWorker(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.s = state
        self.running = True
        self.stop_event = threading.Event()
        self.cap = None
        self.fist_frames = 0

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()

    def run(self):
        face = None
        hands = None
        cap = None
        try:
            ensure_model(FACE_MODEL, FACE_MODEL_URL)
            ensure_model(HAND_MODEL, HAND_MODEL_URL)
            if self.stop_event.is_set():
                return

            face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
            hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))

            cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap = cap
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            if not cap.isOpened():
                print("Could not open camera. Keyboard controls still work.")
                return
            with self.s.lock:
                self.s.camera_ok = True

            t0 = time.monotonic()
            last_ts = -1
            last_t = time.monotonic()
            while self.running and not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.stop_event.wait(0.01):
                        break
                    continue
                frame = cv2.flip(frame, 1)  # mirror: your right = screen right
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = int((time.monotonic() - t0) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                fr = face.detect_for_video(img, ts)
                hr = hands.detect_for_video(img, ts)

                preview = frame.copy()
                face_found = bool(fr.face_landmarks)
                mouth_ratio, head, roll = 0.0, None, 0.0
                if face_found:
                    lm = fr.face_landmarks[0]
                    face_h = dist(lm[10], lm[152]) + 1e-6
                    mouth_ratio = dist(lm[13], lm[14]) / face_h
                    # head tilt (roll) from outer eye corners, in the mirrored image
                    ea, eb = lm[33], lm[263]
                    left_eye, right_eye = (ea, eb) if ea.x < eb.x else (eb, ea)
                    roll = math.degrees(math.atan2((right_eye.y - left_eye.y) * h,
                                                   (right_eye.x - left_eye.x) * w))
                    # roll < 0: head tilted toward screen-left (your left in the mirror)
                    cv2.line(preview, (int(left_eye.x * w), int(left_eye.y * h)),
                             (int(right_eye.x * w), int(right_eye.y * h)), (0, 255, 255), 2)
                    head = self.cut_head(frame, lm, w, h)
                    pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in FACE_OVAL], np.int32)
                    cv2.polylines(preview, [pts], True, (0, 255, 0), 2)

                hand_found = bool(hr.hand_landmarks)
                fist = False
                if hand_found:
                    hl = hr.hand_landmarks[0]
                    wrist = hl[0]
                    curled = sum(1 for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
                                 if dist(hl[tip], wrist) < dist(hl[pip], wrist))
                    self.fist_frames = self.fist_frames + 1 if curled >= 3 else 0
                    fist = self.fist_frames >= 2
                    for p in hl:
                        cv2.circle(preview, (int(p.x * w), int(p.y * h)), 3,
                                   (0, 0, 255) if fist else (255, 200, 0), -1)
                else:
                    self.fist_frames = 0

                small = cv2.resize(preview, (224, 168))
                small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                now = time.monotonic()
                fps = 1.0 / max(now - last_t, 1e-3)
                last_t = now
                with self.s.lock:
                    self.s.face_found = face_found
                    self.s.mouth_ratio = mouth_ratio
                    if face_found:
                        self.s.head_roll = self.s.head_roll * 0.5 + roll * 0.5  # smoothing
                    self.s.hand_found = hand_found
                    self.s.fist = fist
                    if head is not None:
                        self.s.head_rgba = head
                    self.s.preview = (small.tobytes(), 224, 168)
                    self.s.fps = self.s.fps * 0.9 + fps * 0.1
        except Exception as error:
            print(f"Camera tracking stopped: {error}")
        finally:
            self.running = False
            if cap is not None:
                cap.release()
            self.cap = None
            for tracker in (face, hands):
                if tracker is not None:
                    try:
                        tracker.close()
                    except Exception as error:
                        print(f"Could not close MediaPipe tracker: {error}")

    @staticmethod
    def cut_head(frame, lm, w, h):
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in FACE_OVAL], np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad = 0.08 * (x1 - x0)
        x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad * 2))
        x1, y1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        crop = frame[y0:y1, x0:x1]
        mask = np.zeros(crop.shape[:2], np.uint8)
        local = (pts - [x0, y0]).astype(np.int32)
        hull = cv2.convexHull(local)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = mask
        ch, cw = rgba.shape[:2]
        scale = HEAD_SIZE / max(ch, cw)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        rgba = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
        return (rgba.tobytes(), nw, nh)


# ------------------------------------------------------------
# GESTURE INPUT  (turns raw vision data into game actions)
# ------------------------------------------------------------
class GestureInput:
    def __init__(self, state):
        self.s = state
        self.calib_samples = []
        self.roll_samples = []
        self.roll_base = 0.0
        self.tilt = 0.0
        self.calibrating = True
        self.calib_start = time.monotonic()
        self.threshold = MOUTH_MIN_THRESHOLD + 0.02
        self.mouth_open = False
        self.prev_mouth = False
        self.prev_fist = False
        self.jump_pressed = False   # rising edge this frame
        self.jump_held = False
        self.punch_pressed = False
        self.move = 0
        self.speed = 0.0
        self.head_surface = None
        self.preview_surface = None
        self._head_bytes = None

    def recalibrate(self):
        self.calib_samples = []
        self.roll_samples = []
        self.calibrating = True
        self.calib_start = time.monotonic()

    def update(self):
        with self.s.lock:
            face, ratio = self.s.face_found, self.s.mouth_ratio
            hand, fist, roll = self.s.hand_found, self.s.fist, self.s.head_roll
            head, prev = self.s.head_rgba, self.s.preview

        if head is not None and head[0] is not self._head_bytes:
            self._head_bytes = head[0]
            self.head_surface = pygame.image.frombuffer(head[0], (head[1], head[2]), "RGBA").convert_alpha()
        if prev is not None:
            self.preview_surface = pygame.image.frombuffer(prev[0], (prev[1], prev[2]), "RGB")

        if self.calibrating:
            if face:
                self.calib_samples.append(ratio)
                self.roll_samples.append(roll)
            if time.monotonic() - self.calib_start > 2.0 and len(self.calib_samples) >= 10:
                base = float(np.median(self.calib_samples))
                self.threshold = max(MOUTH_MIN_THRESHOLD, base + MOUTH_EXTRA)
                self.roll_base = float(np.median(self.roll_samples))
                self.calibrating = False

        if face and not self.calibrating:
            if self.mouth_open:
                self.mouth_open = ratio > self.threshold * 0.75   # hysteresis
            else:
                self.mouth_open = ratio > self.threshold
        else:
            self.mouth_open = False

        self.jump_pressed = self.mouth_open and not self.prev_mouth
        self.jump_held = self.mouth_open
        self.prev_mouth = self.mouth_open

        self.punch_pressed = fist and not self.prev_fist
        self.prev_fist = fist

        # head tilt -> walking direction and speed (analog)
        self.tilt = roll - self.roll_base if face and not self.calibrating else 0.0
        limit = TILT_RELEASE if self.move != 0 else TILT_DEADZONE
        if abs(self.tilt) < limit or (self.move != 0 and self.tilt * self.move < TILT_RELEASE):
            self.move = 0
        else:
            self.move = -1 if self.tilt < 0 else 1
        if self.move:
            k = (abs(self.tilt) - TILT_RELEASE) / (TILT_FULL - TILT_RELEASE)
            self.speed = max(0.45, min(1.0, k))
        else:
            self.speed = 0.0
        self.hand, self.face, self.fist, self.ratio = hand, face, fist, ratio


# ------------------------------------------------------------
# LEVEL
# ------------------------------------------------------------
class Level:
    def __init__(self, seed=3, length=230):
        rng = random.Random(seed)
        self.length = length
        self.tiles = {}   # (cx, cy) -> 'ground' | 'brick' | 'q' | 'used' | 'pipe'
        self.coins = []
        self.power_boxes = []      # (cx, cy) of gold ? boxes holding a power-up
        self.enemy_spawns = []
        x = 0
        while x < length:
            if 18 < x < length - 20 and rng.random() < 0.07:
                x += rng.randint(2, 3)
                continue
            self.tiles[(x, ROWS - 1)] = "ground"
            self.tiles[(x, ROWS - 2)] = "ground"
            x += 1

        x = 14
        while x < length - 20:
            kind = rng.random()
            if kind < 0.35:          # floating platform with ? block and coins
                y = rng.choice((ROWS - 5, ROWS - 6))
                width = rng.randint(3, 6)
                for i in range(width):
                    self.tiles[(x + i, y)] = "q" if i == width // 2 else "brick"
                if rng.random() < 0.55:      # half of them hold a power-up
                    self.power_boxes.append((x + width // 2, y))
                for i in range(width):
                    self.coins.append(pygame.Rect((x + i) * TILE + 14, (y - 1) * TILE + 10, 20, 28))
            elif kind < 0.55 and (x, ROWS - 2) in self.tiles and (x + 1, ROWS - 2) in self.tiles:
                hgt = rng.randint(1, 2)  # pipe
                for i in range(2):
                    for j in range(hgt):
                        self.tiles[(x + i, ROWS - 3 - j)] = "pipe"
            elif kind < 0.7:         # staircase
                if all((x + i, ROWS - 2) in self.tiles for i in range(4)):
                    for i in range(4):
                        for j in range(i + 1):
                            self.tiles[(x + i, ROWS - 3 - j)] = "brick"
            x += rng.randint(7, 12)

        for cx in range(20, length - 15, 1):
            if (cx, ROWS - 2) in self.tiles and (cx, ROWS - 3) not in self.tiles and rng.random() < 0.30:
                r = rng.random()
                if r < 0.42:
                    kind = "basic"
                elif r < 0.68:
                    kind = "punch"
                elif r < 0.86:
                    kind = "runner"
                else:
                    kind = "brawler"
                _, eh = {"basic": (0, 40), "punch": (0, 58), "runner": (0, 54), "brawler": (0, 62)}[kind]
                self.enemy_spawns.append((cx * TILE, (ROWS - 3) * TILE + 8))
        # one boss near the end of the course, guarding the flag
        bcx = length - 20
        if (bcx, ROWS - 2) in self.tiles:
            self.enemy_spawns.append((bcx * TILE, (ROWS - 3) * TILE - 52))

        self.flag_x = (length - 8) * TILE
        for cx in range(length - 12, length):
            self.tiles[(cx, ROWS - 1)] = "ground"
            self.tiles[(cx, ROWS - 2)] = "ground"

    def solid(self, cx, cy):
        return (cx, cy) in self.tiles

    def rects_near(self, rect):
        x0, x1 = rect.left // TILE - 1, rect.right // TILE + 1
        y0, y1 = rect.top // TILE - 1, rect.bottom // TILE + 1
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                if (cx, cy) in self.tiles:
                    yield (cx, cy), pygame.Rect(cx * TILE, cy * TILE, TILE, TILE)


def move_and_collide(body, level, dt):
    """Axis-separated tile collision. body needs x,y,vx,vy,w,h. Returns (hit_wall, on_ground, head_tiles)."""
    hit_wall, on_ground, head_hits = False, False, []
    body.x += body.vx * dt
    r = body.rect()
    for _, t in level.rects_near(r):
        if r.colliderect(t):
            if body.vx > 0:
                body.x = t.left - body.w
            elif body.vx < 0:
                body.x = t.right
            body.vx = 0
            hit_wall = True
            r = body.rect()
    body.y += body.vy * dt
    r = body.rect()
    for key, t in level.rects_near(r):
        if r.colliderect(t):
            if body.vy > 0:
                body.y = t.top - body.h
                on_ground = True
            elif body.vy < 0:
                body.y = t.bottom
                head_hits.append(key)
            body.vy = 0
            r = body.rect()
    return hit_wall, on_ground, head_hits


# ------------------------------------------------------------
# ENTITIES
# ------------------------------------------------------------
class Particle:
    def __init__(self, x, y, color, vx=None, vy=None, life=0.6, size=5):
        self.x, self.y = x, y
        self.vx = vx if vx is not None else random.uniform(-200, 200)
        self.vy = vy if vy is not None else random.uniform(-450, -150)
        self.life, self.max_life, self.color, self.size = life, life, color, size

    def update(self, dt):
        self.vy += GRAVITY * 0.6 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt


class FloatText:
    def __init__(self, x, y, text):
        self.x, self.y, self.text, self.life = x, y, text, 0.8


class PowerUp:
    """A power-up sitting on the ground, waiting to be picked up."""
    W, H = 40, 40

    def __init__(self, x, y, kind):
        self.x, self.y = float(x), float(y)
        self.kind = kind
        self.vy = -240.0
        self.timer = 0.0

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.W, self.H)

    def update(self, dt, level):
        self.timer += dt
        self.vy = min(self.vy + GRAVITY * dt, MAX_FALL)
        y0 = self.y
        self.y += self.vy * dt
        r = self.rect()
        for _, t in level.rects_near(r):
            if r.colliderect(t):
                if self.vy > 0:
                    self.y = t.top - self.H
                    self.vy = 0
                r = self.rect()

    def draw(self, surf, cam):
        x, y = int(self.x - cam), int(self.y)
        bob = int(math.sin(self.timer * 5) * 3)
        y += bob
        c = {"fire": (240, 80, 40), "wizard": (140, 80, 220), "shield": (60, 180, 240),
             "superjump": (90, 210, 90), "star": (255, 220, 60)}[self.kind]
        pygame.draw.circle(surf, (255, 255, 255), (x + 20, y + 20), 19)
        pygame.draw.circle(surf, c, (x + 20, y + 20), 15)
        if self.kind == "fire":
            for i in range(5):
                a = i * 1.256 + self.timer * 3
                pygame.draw.circle(surf, (255, 220, 120),
                                   (int(x + 20 + math.cos(a) * 7), int(y + 20 + math.sin(a) * 7)), 3)
        elif self.kind == "wizard":
            pygame.draw.polygon(surf, (255, 255, 255), [(x + 20, y + 8), (x + 30, y + 32), (x + 10, y + 32)])
        elif self.kind == "shield":
            pygame.draw.arc(surf, (255, 255, 255), (x + 10, y + 10, 20, 20), 0.6, 5.7, 3)
        elif self.kind == "superjump":
            pygame.draw.polygon(surf, (255, 255, 255), [(x + 20, y + 7), (x + 28, y + 20), (x + 12, y + 20)])
            pygame.draw.polygon(surf, (255, 255, 255), [(x + 20, y + 16), (x + 28, y + 30), (x + 12, y + 30)])
        else:
            pygame.draw.polygon(surf, (255, 255, 255),
                                [(x + 20, y + 5), (x + 25, y + 16), (x + 35, y + 20), (x + 25, y + 24),
                                 (x + 20, y + 35), (x + 15, y + 24), (x + 5, y + 20), (x + 15, y + 16)])


class Projectile:
    """Fireball (burns one enemy) or wizard shot (clears the row)."""

    def __init__(self, x, y, direction, kind):
        self.x, self.y = float(x), float(y)
        self.dir = direction
        self.kind = kind
        self.dead = False
        self.t = 0.0
        self.w = self.h = 18 if kind == "fire" else 22

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def update(self, dt, level):
        self.t += dt
        self.x += self.dir * (FIREBALL_SPEED if self.kind == "fire" else WIZARD_SPEED) * dt
        self.y += math.sin(self.t * 12) * 12 * dt
        r = self.rect()
        for _, t in level.rects_near(r):
            if r.colliderect(t):
                self.dead = True
                return
        if self.t > 2.2:
            self.dead = True

    def draw(self, surf, cam):
        x, y = int(self.x - cam), int(self.y)
        cx, cy = x + self.w // 2, y + self.h // 2
        if self.kind == "fire":
            pygame.draw.circle(surf, (255, 140, 40), (cx, cy), 9)
            pygame.draw.circle(surf, (255, 240, 150), (cx, cy), 5)
            pygame.draw.circle(surf, (255, 90, 20), (cx - self.dir * 8, cy), 6)
            for i in range(4):
                pygame.draw.circle(surf, (255, 180, 60),
                                   (int(cx - self.dir * (10 + i * 4)), int(cy + math.sin(self.t * 20 + i) * 4)), 3)
        else:
            pygame.draw.circle(surf, (200, 160, 255), (cx, cy), 11, 3)
            pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 5)
            for i in range(6):
                a = i * 1.047 + self.t * 8
                pygame.draw.circle(surf, (170, 120, 255),
                                   (int(cx + math.cos(a) * 13), int(cy + math.sin(a) * 13)), 3)


class Enemy:
    def __init__(self, x, y, kind="basic"):
        self.kind = kind
        sizes = {"basic": (40, 40), "punch": (46, 58), "runner": (44, 54),
                 "brawler": (54, 62), "boss": (86, 92)}
        hp = {"basic": 1, "punch": 1, "runner": 1, "brawler": 2, "boss": 4}
        speed = {"basic": 80.0, "punch": 95.0, "runner": 165.0, "brawler": 70.0, "boss": 50.0}
        self.w, self.h = sizes[kind]
        self.hp = hp[kind]
        self.max_hp = self.hp
        self.speed = speed[kind]
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = -self.speed, 0.0
        self.alive = True
        self.dead_timer = 0.0
        self.squashed = False
        self.anim = random.random() * 10
        self.active = False
        self.hurt_flash = 0.0

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def update(self, dt, level):
        self.anim += dt
        self.hurt_flash = max(0.0, self.hurt_flash - dt)
        if not self.alive:
            self.dead_timer += dt
            if not self.squashed:
                self.vy += GRAVITY * dt
                self.x += self.vx * dt
                self.y += self.vy * dt
            return
        self.vy = min(self.vy + GRAVITY * dt, MAX_FALL)
        old_vx = self.vx
        hit_wall, on_ground, _ = move_and_collide(self, level, dt)
        if self.vx == 0:
            self.vx = old_vx
        if hit_wall:
            self.vx = -old_vx
        elif on_ground:
            ahead = self.x + (self.w + 2 if self.vx > 0 else -2)
            if not level.solid(int(ahead // TILE), int((self.y + self.h + 4) // TILE)):
                self.vx = -self.vx
        if self.y > SCREEN_H + 200:
            self.alive = False
            self.dead_timer = 99

    def damage(self, direction=1, squash=False):
        """Returns True when this hit finishes the enemy off."""
        self.hp -= 1
        self.hurt_flash = 0.18
        if self.hp > 0:
            self.vx = 120 * direction
            return False
        self.kill(squash, direction)
        return True

    def kill(self, squash, direction=1):
        self.alive = False
        self.squashed = squash
        if not squash:
            self.vx = 260 * direction
            self.vy = -520

    def draw(self, surf, cam, sprites=None, t=0.0):
        x, y = int(self.x - cam), int(self.y)
        if self.hurt_flash > 0:
            pass
        if sprites and self.kind in sprites and not self.squashed:
            img = sprites[self.kind]
            bob = int(math.sin(self.anim * 9) * 2)
            r = img.get_rect(midbottom=(x + self.w // 2, y + self.h + bob))
            if self.alive and self.vx > 0:
                img = pygame.transform.flip(img, True, False)
            surf.blit(img, r)
            if self.max_hp > 1:
                self.draw_hp(surf, x, y)
            return
        if self.kind in ("punch", "runner", "brawler", "boss"):
            self.draw_kind(surf, x, y)
            if self.max_hp > 1:
                self.draw_hp(surf, x, y)
            return
        if self.squashed:
            pygame.draw.ellipse(surf, (120, 60, 20), (x, y + 28, self.w, 12))
            return
        step = math.sin(self.anim * 12) * 4 if self.alive else 0
        pygame.draw.ellipse(surf, (40, 20, 10), (x - 2 + step, y + 30, 18, 10))
        pygame.draw.ellipse(surf, (40, 20, 10), (x + 24 - step, y + 30, 18, 10))
        pygame.draw.ellipse(surf, (150, 75, 30), (x, y, self.w, 34))
        pygame.draw.rect(surf, (230, 200, 160), (x + 10, y + 22, 20, 10), border_radius=4)
        for ex in (x + 11, x + 25):
            pygame.draw.ellipse(surf, (255, 255, 255), (ex, y + 8, 8, 12))
            pygame.draw.ellipse(surf, (0, 0, 0), (ex + 2, y + 12, 4, 7))
        pygame.draw.line(surf, (0, 0, 0), (x + 8, y + 6), (x + 18, y + 10), 3)
        pygame.draw.line(surf, (0, 0, 0), (x + 32, y + 6), (x + 22, y + 10), 3)
        if self.max_hp > 1:
            self.draw_hp(surf, x, y)

    def draw_kind(self, surf, x, y):
        """Drawn fallback art for each enemy type (used when no sprite file exists)."""
        w, h = self.w, self.h
        step = math.sin(self.anim * 12) * 4 if self.alive else 0
        if self.kind == "punch":
            pygame.draw.ellipse(surf, (40, 20, 10), (x + 2 + step, y + h - 10, 16, 10))
            pygame.draw.ellipse(surf, (40, 20, 10), (x + w - 18 - step, y + h - 10, 16, 10))
            pygame.draw.ellipse(surf, (90, 50, 30), (x + 2, y, w - 4, h - 16))
            pygame.draw.circle(surf, (240, 150, 120), (x + 4, y + 16), 9)
            pygame.draw.circle(surf, (240, 150, 120), (x + w - 4, y + 16), 9)
            pygame.draw.ellipse(surf, (240, 220, 200), (x + 8, y + 8, w - 16, 18))
            pygame.draw.circle(surf, (30, 30, 30), (x + 15, y + 16), 4)
            pygame.draw.circle(surf, (30, 30, 30), (x + w - 15, y + 16), 4)
            pygame.draw.circle(surf, (200, 40, 40), (x + 15, y + 16), 2)
            pygame.draw.circle(surf, (200, 40, 40), (x + w - 15, y + 16), 2)
        elif self.kind == "runner":
            pygame.draw.ellipse(surf, (40, 20, 10), (x + 4 + step, y + h - 10, 14, 9))
            pygame.draw.ellipse(surf, (40, 20, 10), (x + w - 18 - step, y + h - 10, 14, 9))
            pygame.draw.ellipse(surf, (230, 90, 60), (x + 6, y + 4, w - 12, h - 14))
            pygame.draw.ellipse(surf, (255, 190, 170), (x + 10, y + 12, w - 20, 16))
            pygame.draw.circle(surf, (30, 30, 30), (x + 16, y + 20), 4)
            pygame.draw.circle(surf, (30, 30, 30), (x + w - 16, y + 20), 4)
            pygame.draw.ellipse(surf, (140, 20, 20), (x + 12, y + h - 12, w - 24, 10))
        elif self.kind == "brawler":
            pygame.draw.ellipse(surf, (40, 20, 10), (x + 2 + step, y + h - 10, 18, 10))
            pygame.draw.ellipse(surf, (40, 20, 10), (x + w - 20 - step, y + h - 10, 18, 10))
            pygame.draw.circle(surf, (230, 100, 120), (x + w // 2, y + 24), 22)
            pygame.draw.circle(surf, (255, 200, 180), (x + w // 2, y + 20), 15)
            pygame.draw.circle(surf, (220, 60, 90), (x + 6, y + 30), 11)
            pygame.draw.circle(surf, (220, 60, 90), (x + w - 6, y + 30), 11)
            pygame.draw.circle(surf, (255, 190, 190), (x + 6, y + 27), 5)
            pygame.draw.circle(surf, (255, 190, 190), (x + w - 6, y + 27), 5)
            pygame.draw.circle(surf, (40, 30, 30), (x + w // 2 - 6, y + 18), 4)
            pygame.draw.circle(surf, (40, 30, 30), (x + w // 2 + 6, y + 18), 4)
        else:  # boss
            pygame.draw.ellipse(surf, (40, 20, 10), (x + 4 + step, y + h - 12, 24, 12))
            pygame.draw.ellipse(surf, (40, 20, 10), (x + w - 28 - step, y + h - 12, 24, 12))
            pygame.draw.rect(surf, (215, 200, 200), (x + 6, y + 16, w - 12, h - 22), border_radius=8)
            for i in range(5):
                px_ = x + 8 + i * 18
                pygame.draw.rect(surf, (120, 120, 120), (px_, y + 20 + (i % 2) * 12, 6, 6))
            pygame.draw.circle(surf, (255, 255, 255), (x + 18, y + 26), 12)
            pygame.draw.circle(surf, (255, 255, 255), (x + w - 18, y + 26), 12)
            pygame.draw.circle(surf, (30, 30, 30), (x + 18, y + 28), 6)
            pygame.draw.circle(surf, (30, 30, 30), (x + w - 18, y + 28), 6)
            pygame.draw.ellipse(surf, (255, 140, 120), (x + 14, y + 30, w - 28, 22))
            pygame.draw.ellipse(surf, (120, 20, 20), (x + 24, y + 38, w - 48, 12))
            pygame.draw.circle(surf, (75, 180, 235), (x + 24, y + 24), 8, 3)
            pygame.draw.circle(surf, (75, 180, 235), (x + w - 24, y + 24), 8, 3)

    def draw_hp(self, surf, x, y):
        w = self.w
        pygame.draw.rect(surf, (0, 0, 0), (x, y - 10, w, 6))
        pygame.draw.rect(surf, (240, 60, 60), (x + 1, y - 9, int((w - 2) * self.hp / self.max_hp), 4))


class Player:
    def __init__(self, x, y):
        self.w, self.h = 34, 62
        self.reset(x, y)

    def reset(self, x, y):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.on_ground = False
        self.facing = 1
        self.walk_phase = 0.0
        self.coyote = 0.0
        self.jump_buffer = 0.0
        self.punch_timer = 0.0
        self.punch_cd = 0.0
        self.punch_arm = 1
        self.invincible = 0.0
        self.dead = False
        self.jumping = False
        self.star_mode = False

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def punch_rect(self):
        if self.punch_timer <= 0:
            return None
        if self.facing > 0:
            return pygame.Rect(int(self.x + self.w - 4), int(self.y + 14), PUNCH_RANGE, 30)
        return pygame.Rect(int(self.x - PUNCH_RANGE + 4), int(self.y + 14), PUNCH_RANGE, 30)

    def update(
        self, dt, level, move, jump_pressed, jump_held, punch_pressed, sounds,
        speed=1.0, super_jump=False
    ):
        if move:
            self.facing = move
            top = WALK_SPEED * speed
            self.vx += move * ACCEL * dt
            if abs(self.vx) > top:
                self.vx = max(-top, min(top, self.vx - move * FRICTION * dt)) if self.vx * move > 0 else self.vx
        else:
            if self.vx > 0:
                self.vx = max(0, self.vx - FRICTION * dt)
            elif self.vx < 0:
                self.vx = min(0, self.vx + FRICTION * dt)

        self.coyote = COYOTE_TIME if self.on_ground else self.coyote - dt
        self.jump_buffer = JUMP_BUFFER if jump_pressed else self.jump_buffer - dt
        if self.jump_buffer > 0 and self.coyote > 0:
            self.vy = -SUPER_JUMP_VEL if super_jump else -JUMP_VEL
            self.jump_buffer = self.coyote = 0
            self.jumping = True
            sounds.play("jump")
        if self.jumping and not jump_held and self.vy < 0:
            self.vy *= JUMP_CUT
            self.jumping = False

        self.punch_cd -= dt
        self.punch_timer -= dt
        if punch_pressed and self.punch_cd <= 0:
            self.punch_timer = PUNCH_TIME
            self.punch_cd = PUNCH_COOLDOWN
            self.punch_arm *= -1
            sounds.play("punch")

        self.vy = min(self.vy + GRAVITY * dt, MAX_FALL)
        _, self.on_ground, head_hits = move_and_collide(self, level, dt)
        if self.on_ground:
            self.jumping = False
        if self.x < 0:
            self.x, self.vx = 0, 0
        if abs(self.vx) > 10 and self.on_ground:
            self.walk_phase += abs(self.vx) * dt * 0.045
        else:
            self.walk_phase *= 0.8
        self.invincible -= dt
        return head_hits

    def draw(self, surf, cam, head_surface, t):
        if self.invincible > 0 and int(t * 20) % 2 == 0:
            return
        star = self.star_mode
        x, y = int(self.x - cam), int(self.y)
        cx = x + self.w // 2
        f = self.facing
        swing = math.sin(self.walk_phase) * 0.9 if self.on_ground else 0.5
        hip_y = y + 46
        leg_len = 16
        for side in (1, -1):
            ang = swing * side if self.on_ground else (0.6 if side == 1 else -0.3)
            fx = cx + side * 5 + math.sin(ang) * leg_len * f
            fy = hip_y + math.cos(ang) * leg_len
            pygame.draw.line(surf, (30, 60, 200), (cx + side * 5, hip_y), (fx, fy), 9)
            pygame.draw.ellipse(surf, (90, 50, 20), (fx - 8 + 3 * f, fy - 3, 16, 8))
        body = (255, 220, 60) if star else (210, 30, 30)
        pygame.draw.rect(surf, body, (x + 3, y + 22, self.w - 6, 20), border_radius=6)
        pygame.draw.rect(surf, (30, 60, 200), (x + 5, y + 32, self.w - 10, 16), border_radius=4)
        pygame.draw.circle(surf, (255, 220, 0), (x + 11, y + 35), 3)
        pygame.draw.circle(surf, (255, 220, 0), (x + self.w - 11, y + 35), 3)

        shoulder = (cx, y + 27)
        punching = self.punch_timer > 0
        ext = 0
        if punching:
            p = 1 - self.punch_timer / PUNCH_TIME
            ext = math.sin(min(p, 1) * math.pi)
        back_ang = -swing * 0.8
        bx = shoulder[0] - f * 4 + math.sin(back_ang) * 14 * f
        by = shoulder[1] + math.cos(back_ang) * 14
        pygame.draw.line(surf, body, shoulder, (bx, by), 7)
        pygame.draw.circle(surf, (255, 255, 255), (int(bx), int(by)), 6)
        if punching:
            hx = shoulder[0] + f * (10 + ext * 34)
            hy = shoulder[1] - 2
            pygame.draw.line(surf, body, shoulder, (hx, hy), 8)
            pygame.draw.circle(surf, (255, 255, 255), (int(hx), int(hy)), 10)
            pygame.draw.circle(surf, (200, 200, 200), (int(hx), int(hy)), 10, 2)
            if ext > 0.7:
                for i in range(3):
                    lx = hx + f * 14
                    pygame.draw.line(surf, (255, 240, 120), (lx, hy - 8 + i * 8), (lx + f * 12, hy - 10 + i * 10), 2)
        else:
            fa = swing * 0.8
            fx2 = shoulder[0] + f * 4 + math.sin(fa) * 14 * f
            fy2 = shoulder[1] + math.cos(fa) * 14
            pygame.draw.line(surf, body, shoulder, (fx2, fy2), 7)
            pygame.draw.circle(surf, (255, 255, 255), (int(fx2), int(fy2)), 6)

        head_cx, head_cy = cx, y + 4
        if head_surface is not None:
            bob = int(abs(math.sin(self.walk_phase)) * 2)
            r = head_surface.get_rect(center=(head_cx, head_cy - bob))
            surf.blit(head_surface, r)
            pygame.draw.ellipse(surf, (210, 30, 30), (head_cx - 26, r.top - 8, 52, 16))
            pygame.draw.rect(surf, (210, 30, 30), (head_cx - 18, r.top - 16, 36, 12), border_radius=8)
            pygame.draw.circle(surf, (255, 255, 255), (head_cx, r.top - 9), 5)
        else:
            pygame.draw.circle(surf, (255, 205, 160), (head_cx, head_cy), 22)
            pygame.draw.circle(surf, (0, 0, 0), (head_cx + 8 * f, head_cy - 4), 3)
            pygame.draw.rect(surf, (80, 40, 10), (head_cx - 4 + 6 * f, head_cy + 6, 14, 4))
            pygame.draw.ellipse(surf, (210, 30, 30), (head_cx - 24, head_cy - 26, 48, 18))


# ------------------------------------------------------------
# SOUND (generated, no files needed)
# ------------------------------------------------------------
class Sounds:
    def __init__(self):
        self.sounds = {}
        try:
            pygame.mixer.init(44100, -16, 1, 512)
            self.sounds = {
                "jump": self.tone([(520, 0.04), (780, 0.08)]),
                "punch": self.noise(0.08),
                "coin": self.tone([(988, 0.05), (1319, 0.15)]),
                "stomp": self.tone([(300, 0.04), (180, 0.08)]),
                "hurt": self.tone([(400, 0.1), (300, 0.1), (200, 0.2)]),
                "win": self.tone([(523, 0.1), (659, 0.1), (784, 0.1), (1047, 0.3)]),
            }
        except Exception:
            pass

    @staticmethod
    def tone(notes, vol=0.25):
        sr = 44100
        parts = []
        for freq, dur in notes:
            t = np.arange(int(sr * dur)) / sr
            wave = np.sign(np.sin(2 * np.pi * freq * t)) * np.linspace(1, 0.2, t.size)
            parts.append(wave)
        data = (np.concatenate(parts) * vol * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(data)

    @staticmethod
    def noise(dur, vol=0.3):
        n = int(44100 * dur)
        data = (np.random.uniform(-1, 1, n) * np.linspace(1, 0, n) * vol * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(data)

    def play(self, name):
        s = self.sounds.get(name)
        if s:
            s.play()


# ------------------------------------------------------------
# GAME
# ------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Face Mario")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22, bold=True)
        self.big = pygame.font.SysFont("arial", 56, bold=True)
        self.small = pygame.font.SysFont("arial", 15)
        self.sounds = Sounds()
        self.background = self.load_background()
        self.tile_cache = self.make_tiles()
        self.sprites = self.load_sprites()

        self.vstate = VisionState()
        self.worker = VisionWorker(self.vstate)
        self.worker.start()
        self.gest = GestureInput(self.vstate)
        self.lives = 3
        self.score = 0
        self.coins = 0
        self.new_level()

    def load_sprites(self):
        """Load the enemy pictures from sprites/. Falls back to drawn art."""
        out = {}
        wanted = {"enemy_punch": 46, "enemy_brawler": 54, "enemy_runner": 44, "enemy_boss": 86}
        for name, size in wanted.items():
            path = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                cand = SPRITE_DIR / (name + ext)
                if cand.exists():
                    path = cand
                    break
            if path is None:
                continue
            try:
                img = pygame.image.load(str(path)).convert_alpha()
                big = max(img.get_width(), img.get_height())
                if big > 0:
                    k = size / big
                    img = pygame.transform.smoothscale(
                        img, (max(1, int(img.get_width() * k)), max(1, int(img.get_height() * k))))
                out[name] = img
            except Exception as e:
                print("Could not load", path, e)
        if out:
            print("Loaded sprites:", ", ".join(sorted(out)))
        return out

    def load_background(self):
        if BACKGROUND_FILE.exists():
            img = pygame.image.load(str(BACKGROUND_FILE)).convert()
            scale = SCREEN_H / img.get_height()
            return pygame.transform.smoothscale(img, (int(img.get_width() * scale), SCREEN_H))
        bg = pygame.Surface((SCREEN_W, SCREEN_H))
        for yy in range(SCREEN_H):
            k = yy / SCREEN_H
            bg.fill((int(92 + 60 * k), int(148 + 60 * k), 252), (0, yy, SCREEN_W, 1))
        for i in range(6):
            cx = 80 + i * 170
            cy = 70 + (i % 3) * 40
            for dx, r in ((0, 26), (26, 34), (56, 26)):
                pygame.draw.circle(bg, (255, 255, 255), (cx + dx, cy), r)
        for i in range(4):
            pygame.draw.circle(bg, (60, 170, 70), (120 + i * 260, SCREEN_H - 60), 120)
        return bg

    def make_tiles(self):
        tiles = {}
        g = pygame.Surface((TILE, TILE)); g.fill((200, 110, 50))
        for i in range(0, TILE, 12):
            pygame.draw.line(g, (150, 75, 30), (0, i), (TILE, i), 2)
        tiles["ground"] = g
        gt = g.copy(); pygame.draw.rect(gt, (70, 190, 60), (0, 0, TILE, 12))
        pygame.draw.rect(gt, (40, 150, 40), (0, 10, TILE, 3))
        tiles["grass"] = gt
        b = pygame.Surface((TILE, TILE)); b.fill((180, 80, 40))
        for row in range(4):
            yy = row * 12
            pygame.draw.line(b, (90, 40, 20), (0, yy), (TILE, yy), 2)
            off = 0 if row % 2 == 0 else TILE // 2
            for xx in range(off, TILE + 1, TILE // 2):
                pygame.draw.line(b, (90, 40, 20), (xx, yy), (xx, yy + 12), 2)
        tiles["brick"] = b
        q = pygame.Surface((TILE, TILE)); q.fill((250, 180, 30))
        pygame.draw.rect(q, (150, 80, 0), q.get_rect(), 3)
        txt = self.font.render("?", True, (120, 50, 0)) if hasattr(self, "font") else None
        if txt:
            q.blit(txt, txt.get_rect(center=(TILE // 2, TILE // 2)))
        tiles["q"] = q
        u = pygame.Surface((TILE, TILE)); u.fill((140, 100, 60))
        pygame.draw.rect(u, (80, 50, 30), u.get_rect(), 3)
        tiles["used"] = u
        p = pygame.Surface((TILE, TILE)); p.fill((40, 180, 60))
        pygame.draw.rect(p, (20, 110, 30), (0, 0, 8, TILE)); pygame.draw.rect(p, (120, 230, 120), (12, 0, 6, TILE))
        tiles["pipe"] = p
        pt = p.copy(); pygame.draw.rect(pt, (20, 110, 30), (0, 0, TILE, 10)); tiles["pipe_top"] = pt
        return tiles

    def new_level(self):
        self.level = Level(seed=random.randint(0, 9999))
        self.player = Player(3 * TILE, (ROWS - 4) * TILE)
        self.enemies = []
        for x, y in self.level.enemy_spawns:
            boss = abs(x - (self.level.length - 20) * TILE) < TILE
            self.enemies.append(Enemy(x, y, "boss" if boss else
                                      random.choice(["basic", "basic", "punch", "runner", "brawler"])))
        self.projectiles, self.powerups = [], []
        self.active_power = None
        self.power_timer = 0.0
        self.shield = False
        self.particles, self.texts = [], []
        self.cam = 0.0
        self.state = "play"
        self.time_left = 300.0
        self.paused = False
        self.state_timer = 0.0

    def give_power(self, kind):
        self.active_power = kind
        self.power_timer = POWER_DURATION.get(kind, 15.0)
        name = POWER_NAMES[kind]
        self.texts.append(FloatText(self.player.x, self.player.y - 40, name))
        self.sounds.play("coin")
        self.burst(self.player.x + 17, self.player.y + 20, (255, 240, 150), 18)
        if kind == "shield":
            self.shield = True
        print("Power:", name)

    def spawn_powerup(self, cx, cy):
        kind = random.choice(["fire", "wizard", "shield", "superjump", "star"])
        self.powerups.append(PowerUp(cx * TILE + 4, cy * TILE - 8, kind))
        self.burst(cx * TILE + 24, cy * TILE, (255, 255, 180), 12)

    def power_active(self, kind):
        return self.active_power == kind and self.power_timer > 0

    def respawn(self):
        self.player.reset(max(3 * TILE, self.cam + 3 * TILE), (ROWS - 6) * TILE)
        while not self.level.solid(int((self.player.x + 17) // TILE), ROWS - 2):
            self.player.x += TILE
        self.player.invincible = INVINCIBLE_TIME
        self.active_power = None
        self.power_timer = 0.0
        self.shield = False
        self.projectiles = []
        self.state = "play"

    def hurt(self):
        if self.player.invincible > 0 or self.state != "play":
            return
        if self.shield:
            self.shield = False
            self.active_power = None
            self.power_timer = 0.0
            self.player.invincible = INVINCIBLE_TIME
            self.sounds.play("stomp")
            self.texts.append(FloatText(self.player.x, self.player.y - 30, "SHIELD BLOCKED!"))
            self.burst(self.player.x + 17, self.player.y + 20, (120, 220, 255), 16)
            return
        self.sounds.play("hurt")
        self.lives -= 1
        self.player.dead = True
        self.player.vy = -700
        self.state = "dying"
        self.state_timer = 0

    def burst(self, x, y, color, n=12):
        for _ in range(n):
            self.particles.append(Particle(x, y, color))

    # --------------------------------------------------------
    def update(self, dt):
        g = self.gest
        g.update()
        keys = pygame.key.get_pressed()
        move = g.move
        speed = g.speed if g.move else 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            move, speed = -1, 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            move, speed = 1, 1.0
        kjump_held = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]
        jump_pressed = g.jump_pressed or self.key_jump
        jump_held = g.jump_held or kjump_held
        punch_pressed = g.punch_pressed or self.key_punch
        self.key_jump = self.key_punch = False

        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.life > 0]
        for t in self.texts:
            t.y -= 60 * dt
            t.life -= dt
        self.texts = [t for t in self.texts if t.life > 0]

        if self.state in ("gameover", "win"):
            self.state_timer += dt
            if self.state_timer > 2.0 and (jump_pressed or punch_pressed):
                if self.state == "gameover":
                    self.lives, self.score, self.coins = 3, 0, 0
                self.new_level()
            return

        if self.state == "dying":
            self.state_timer += dt
            self.player.vy += GRAVITY * dt
            self.player.y += self.player.vy * dt
            if self.state_timer > 1.8:
                if self.lives <= 0:
                    self.state, self.state_timer = "gameover", 0
                else:
                    self.respawn()
            return

        if self.active_power:
            self.power_timer -= dt
            if self.power_timer <= 0:
                self.active_power = None
                self.shield = False

        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 300
            self.hurt()
            return

        pl = self.player
        pl.star_mode = self.power_active("star")
        head_hits = pl.update(
            dt,
            self.level,
            move,
            jump_pressed,
            jump_held,
            punch_pressed,
            self.sounds,
            speed,
            super_jump=self.power_active("superjump"),
        )
        for key in head_hits:
            kind = self.level.tiles.get(key)
            if kind == "q" and key in self.level.power_boxes:
                self.level.power_boxes.remove(key)
                self.level.tiles[key] = "used"
                self.spawn_powerup(key[0], key[1])
                self.sounds.play("jump")
                self.texts.append(FloatText(key[0] * TILE + 10, key[1] * TILE - 28, "POWER!"))
            elif kind == "q":
                self.level.tiles[key] = "used"
                self.coins += 1
                self.score += 200
                self.sounds.play("coin")
                self.texts.append(FloatText(key[0] * TILE + 10, key[1] * TILE - 20, "+200"))
                self.burst(key[0] * TILE + 24, key[1] * TILE, (255, 220, 0), 8)
            elif kind == "brick":
                self.burst(key[0] * TILE + 24, key[1] * TILE + 24, (180, 80, 40), 6)

        if pl.y > SCREEN_H + 50:
            self.hurt()
            return

        prect = pl.rect()
        for c in self.level.coins[:]:
            if prect.colliderect(c):
                self.level.coins.remove(c)
                self.coins += 1
                self.score += 100
                self.sounds.play("coin")
                if self.coins % 100 == 0:
                    self.lives += 1

        # power-ups lying on the ground
        for p in self.powerups[:]:
            p.update(dt, self.level)
            if p.rect().colliderect(prect):
                self.powerups.remove(p)
                self.give_power(p.kind)

        # shots: fireball burns one enemy, wizard clears the row
        for pr in self.projectiles[:]:
            pr.update(dt, self.level)
            if pr.kind == "wizard" and pr.t <= 0.02:
                # the magic shot clears every enemy in front of the player, in its lane
                for e in self.enemies:
                    if not e.alive:
                        continue
                    er = e.rect()
                    ahead = (er.left >= pr.x - 8) if pr.dir > 0 else (er.right <= pr.x + pr.w + 8)
                    close = abs(er.centerx - pl.x) < SCREEN_W * 0.7
                    if ahead and close and abs(er.centery - (pl.y + 20)) < 150:
                        if e.damage(pr.dir):
                            self.score += 300
                            self.texts.append(FloatText(e.x, e.y - 16, "ZAP! +300"))
                        self.burst(e.x + e.w / 2, e.y + e.h / 2, (200, 160, 255), 16)
            elif pr.kind == "fire":
                for e in self.enemies:
                    if e.alive and pr.rect().colliderect(e.rect()):
                        if e.damage(pr.dir):
                            self.score += 300
                            self.texts.append(FloatText(e.x, e.y - 16, "BURN! +300"))
                        self.burst(pr.x, pr.y, (255, 170, 60), 14)
                        pr.dead = True
                        break
            if pr.dead:
                self.projectiles.remove(pr)

        punch = pl.punch_rect()
        swung = pl.punch_timer > PUNCH_TIME * 0.85
        if punch and self.power_active("fire") and swung:
            self.projectiles.append(Projectile(punch.right if pl.facing > 0 else punch.left - 18,
                                               pl.y + 20, pl.facing, "fire"))
        if punch and self.power_active("wizard") and swung:
            self.projectiles.append(Projectile(punch.right if pl.facing > 0 else punch.left - 22,
                                               pl.y + 18, pl.facing, "wizard"))
            self.sounds.play("punch")

        for e in self.enemies:
            if not e.active and e.x < self.cam + SCREEN_W + 100:
                e.active = True
            if not e.active:
                continue
            e.update(dt, self.level)
            if not e.alive:
                continue
            er = e.rect()
            if punch and punch.colliderect(er):
                finished = e.damage(pl.facing)
                self.score += 300
                self.sounds.play("stomp")
                self.texts.append(FloatText(e.x, e.y - 20, ("POW! +300" if finished else "HIT!")))
                self.burst(e.x + e.w / 2, e.y + e.h / 2, (255, 240, 120), 14)
                if not finished:
                    pl.vx = -pl.facing * 180
                continue
            if prect.colliderect(er):
                if self.power_active("star"):
                    e.kill(False, pl.facing)
                    self.score += 300
                    self.sounds.play("stomp")
                    self.burst(e.x + e.w / 2, e.y + e.h / 2, (255, 220, 80), 16)
                    continue
                if pl.vy > 0 and prect.bottom - er.top < 30:
                    e.kill(True)
                    pl.vy = -JUMP_VEL * 0.6
                    self.score += 200
                    self.sounds.play("stomp")
                    self.texts.append(FloatText(e.x, e.y - 20, "+200"))
                else:
                    self.hurt()
                    return
        self.enemies = [e for e in self.enemies if e.alive or e.dead_timer < 1.5]

        if pl.x > self.level.flag_x:
            self.state, self.state_timer = "win", 0
            self.score += int(self.time_left) * 10
            self.sounds.play("win")

        target = pl.x - SCREEN_W * 0.4
        self.cam += (target - self.cam) * min(1, dt * 8)
        self.cam = max(0, min(self.cam, self.level.length * TILE - SCREEN_W))

    # --------------------------------------------------------
    def draw(self, t):
        s = self.screen
        bw = self.background.get_width()
        off = -int(self.cam * 0.3) % bw
        x = off - bw
        while x < SCREEN_W:
            s.blit(self.background, (x, 0))
            x += bw

        cam = int(self.cam)
        c0, c1 = cam // TILE - 1, (cam + SCREEN_W) // TILE + 1
        for (cx, cy), kind in self.level.tiles.items():
            if c0 <= cx <= c1:
                img = kind
                if kind == "ground" and (cx, cy - 1) not in self.level.tiles:
                    img = "grass"
                if kind == "pipe" and self.level.tiles.get((cx, cy - 1)) != "pipe":
                    img = "pipe_top"
                s.blit(self.tile_cache[img], (cx * TILE - cam, cy * TILE))

        for c in self.level.coins:
            if cam - 40 < c.x < cam + SCREEN_W:
                wdt = max(4, int(abs(math.sin(t * 4 + c.x)) * c.w))
                pygame.draw.ellipse(s, (255, 210, 0), (c.centerx - wdt // 2 - cam, c.y, wdt, c.h))
                pygame.draw.ellipse(s, (200, 140, 0), (c.centerx - wdt // 2 - cam, c.y, wdt, c.h), 2)

        fx = self.level.flag_x - cam
        pygame.draw.rect(s, (220, 220, 220), (fx, 2 * TILE, 6, (ROWS - 4) * TILE))
        pygame.draw.circle(s, (255, 215, 0), (fx + 3, 2 * TILE), 9)
        wave = int(math.sin(t * 6) * 4)
        pygame.draw.polygon(s, (40, 200, 60), [(fx + 6, 2 * TILE + 10), (fx + 56, 2 * TILE + 26 + wave), (fx + 6, 2 * TILE + 44)])

        for p in self.powerups:
            if cam - 60 < p.x < cam + SCREEN_W:
                p.draw(s, cam)
        for e in self.enemies:
            if e.active:
                e.draw(s, cam, self.sprites.get(e.kind), t)
        self.player.draw(s, cam, self.gest.head_surface, t)
        for pr in self.projectiles:
            pr.draw(s, cam)

        for p in self.particles:
            pygame.draw.rect(s, p.color, (p.x - cam, p.y, p.size, p.size))
        for ft in self.texts:
            img = self.font.render(ft.text, True, (255, 255, 255))
            s.blit(img, (ft.x - cam, ft.y))

        self.draw_hud(t)

    def text(self, msg, pos, font=None, color=(255, 255, 255), center=False):
        font = font or self.font
        img = font.render(msg, True, color)
        sh = font.render(msg, True, (0, 0, 0))
        r = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
        self.screen.blit(sh, r.move(2, 2))
        self.screen.blit(img, r)

    def draw_hud(self, t):
        g = self.gest
        self.text(f"SCORE {self.score:06d}", (16, 10))
        self.text(f"COINS x{self.coins:02d}", (230, 10))
        self.text(f"LIVES x{self.lives}", (400, 10))
        self.text(f"TIME {int(self.time_left):03d}", (550, 10))
        if self.active_power:
            label = POWER_NAMES[self.active_power]
            bar = self.power_timer / POWER_DURATION.get(self.active_power, 15.0)
            col = {"fire": (255, 150, 60), "wizard": (190, 150, 255), "shield": (120, 220, 255),
                   "superjump": (140, 240, 140), "star": (255, 230, 90)}[self.active_power]
            surf = self.screen
            pygame.draw.rect(surf, (0, 0, 0), (656, 12, 200, 22))
            pygame.draw.rect(surf, col, (658, 14, int(196 * max(0.0, min(1.0, bar))), 18))
            self.text(label, (660, 13), self.small, (20, 20, 20))
            if self.shield:
                self.text("SHIELD READY", (660, 36), self.small, (150, 220, 255))
        pass

        if g.preview_surface is not None:
            px, py = SCREEN_W - 234, 10
            pygame.draw.rect(self.screen, (0, 0, 0), (px - 3, py - 3, 230, 174))
            self.screen.blit(g.preview_surface, (px, py))
            y = py + 172
            mouth = "OPEN (JUMP)" if g.mouth_open else "closed"
            move = {-1: "LEFT", 0: "stop", 1: "RIGHT"}[g.move]
            for line, ok in ((f"Face: {'yes' if g.face else 'NO'}  Mouth: {mouth}", g.face),
                             (f"Tilt: {g.tilt:+.0f} deg  -> {move}", g.face),
                             (f"Hand: {'yes' if g.hand else 'NO'}  {'FIST (PUNCH)' if g.fist else 'open'}", g.hand)):
                self.text(line, (px, y), self.small, (180, 255, 180) if ok else (255, 160, 160))
                y += 18
        elif not self.vstate.camera_ok:
            self.text("Starting camera... (keyboard works)", (SCREEN_W - 330, 10), self.small)

        if g.calibrating:
            self.text("CALIBRATING - head STRAIGHT, mouth CLOSED", (SCREEN_W // 2, 90), self.font, (255, 255, 0), True)

        if self.paused:
            self.text("PAUSED", (SCREEN_W // 2, SCREEN_H // 2), self.big, center=True)
        if self.state == "gameover":
            self.text("GAME OVER", (SCREEN_W // 2, SCREEN_H // 2 - 30), self.big, (255, 80, 80), True)
            if self.state_timer > 2:
                self.text("Open mouth / make a fist / SPACE to restart", (SCREEN_W // 2, SCREEN_H // 2 + 30), center=True)
        if self.state == "win":
            self.text("COURSE CLEAR!", (SCREEN_W // 2, SCREEN_H // 2 - 30), self.big, (255, 230, 80), True)
            if self.state_timer > 2:
                self.text("Open mouth / make a fist / SPACE for next level", (SCREEN_W // 2, SCREEN_H // 2 + 30), center=True)

    # --------------------------------------------------------
    def run(self):
        self.key_jump = self.key_punch = False
        running = True
        try:
            while running:
                dt = min(self.clock.tick(FPS) / 1000.0, 1 / 30)
                t = pygame.time.get_ticks() / 1000.0
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            running = False
                        elif ev.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
                            self.key_jump = True
                        elif ev.key in (pygame.K_f, pygame.K_j):
                            self.key_punch = True
                        elif ev.key == pygame.K_c:
                            self.gest.recalibrate()
                        elif ev.key == pygame.K_p:
                            self.paused = not self.paused
                        elif ev.key == pygame.K_r:
                            self.lives, self.score, self.coins = 3, 0, 0
                            self.new_level()
                if not self.paused:
                    self.update(dt)
                else:
                    self.gest.update()
                self.draw(t)
                pygame.display.flip()
        finally:
            self.worker.stop()
            self.worker.join()
            pygame.quit()


if __name__ == "__main__":
    Game().run()
