from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


class EngineeringImageAnalyzer:
    def __init__(self, llm_client, logger: logging.Logger, tesseract_cmd: str | None = None) -> None:
        self.llm_client = llm_client
        self.logger = logger
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _ocr(self, image_path: Path) -> str:
        try:
            text = pytesseract.image_to_string(Image.open(image_path), config="--psm 6")
            return text.strip()
        except Exception as exc:
            self.logger.warning("OCR failed for %s: %s", image_path, exc)
            return ""

    def _extract_vision_features(self, image_path: Path) -> dict:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": "image_load_failed"}

            edges = cv2.Canny(img, 80, 180)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=25, maxLineGap=8)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            circles = cv2.HoughCircles(
                img,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=20,
                param1=100,
                param2=25,
                minRadius=8,
                maxRadius=120,
            )

            return {
                "shape": [int(img.shape[0]), int(img.shape[1])],
                "edge_density": float(np.mean(edges > 0)),
                "line_count": int(0 if lines is None else len(lines)),
                "contour_count": int(len(contours)),
                "circle_count": int(0 if circles is None else circles.shape[1]),
            }
        except Exception as exc:
            self.logger.warning("Vision feature extraction failed for %s: %s", image_path, exc)
            return {"error": str(exc)}

    def analyze(self, image_path: Path) -> str:
        ocr_text = self._ocr(image_path)
        features = self._extract_vision_features(image_path)

        return self.llm_client.interpret_engineering_image(
            image_path=str(image_path),
            ocr_text=ocr_text,
            vision_features=features,
        )
