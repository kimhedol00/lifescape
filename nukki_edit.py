import cv2
import numpy as np
from PIL import Image
from transparent_background import Remover
import os

class ImageEditor:
    def __init__(self, img_path, mask_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file '{img_path}' not found.")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file '{mask_path}' not found.")

        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Failed to read image from '{img_path}'.")

        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if self.mask is None:
            raise ValueError(f"Failed to read mask from '{mask_path}'.")
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)

        self.drawing = False
        self.brush_size = 20
        self.alpha = 0.5
        self.last_x, self.last_y = None, None

        self.history = []  # To store the history of images for undo functionality

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_circle)

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_x, self.last_y = x, y
            self.save_state()  # Save the state before drawing
            self.apply_brush(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.apply_brush(x, y)
                self.last_x, self.last_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_x, self.last_y = None, None

    def apply_brush(self, x, y):
        self.apply_soft_brush(self.img, self.last_x, self.last_y, x, y, self.brush_size, (0, 0, 0, 0))
        self.update_mask(x, y)

    def apply_soft_brush(self, img, x0, y0, x1, y1, size, color):
        overlay = img.copy()
        if x0 is None or y0 is None:
            cv2.circle(overlay, (x1, y1), size, color, -1)
        else:
            cv2.line(overlay, (x0, y0), (x1, y1), color, size * 2)
            cv2.circle(overlay, (x1, y1), size, color, -1)
        cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0, img)

    def update_mask(self, x, y):
        self.apply_hard_brush(self.mask, self.last_x, self.last_y, x, y, self.brush_size, (0, 0, 0, 0))

    def apply_hard_brush(self, img, x0, y0, x1, y1, size, color):
        if x0 is None or y0 is None:
            cv2.circle(img, (x1, y1), size, color, -1)
        else:
            cv2.line(img, (x0, y0), (x1, y1), color, size * 2)
            cv2.circle(img, (x1, y1), size, color, -1)

    def save_state(self):
        self.history.append((self.img.copy(), self.mask.copy()))

    def undo(self):
        if self.history:
            self.img, self.mask = self.history.pop()

    def run(self):
        while True:
            cv2.imshow('image', self.img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키를 눌러서 종료
                break
            elif key == ord('s'):  # 's' 키를 눌러서 저장
                cv2.imwrite('edited_mask.png', self.mask)
            elif key == ord('['):  # '[' 키를 눌러서 브러쉬 크기 줄이기
                self.brush_size = max(1, self.brush_size - 1)
            elif key == ord(']'):  # ']' 키를 눌러서 브러쉬 크기 키우기
                self.brush_size += 1
            elif key == ord('u'):  # 'u' 키를 눌러서 뒤로 가기
                self.undo()

        cv2.destroyAllWindows()
