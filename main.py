import sys
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QPushButton,
    QLabel,
    QTextEdit,
    QLayout,
    QVBoxLayout,
    QHBoxLayout
)
from PyQt6.QtCore import QTimer, Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Digit Recognition System (IDRS)")
        self.setMinimumSize(QSize(1100,800))

        #Integrating AI model
        try:
            self.model_path = Path(__file__).resolve().parent / "model.keras"
            self.model = load_model(self.model_path)
            print(f"AI model loaded successfully: {self.model_path}")
        except Exception as e:
            self.model = None
            print("Error in loading AI model")
            print(e)

        #Background
        self.setStyleSheet("""
                    QMainWindow {
                        background-color: qlineargradient(
                            x1: 0, y1: 0, 
                            x2: 1, y2: 1, 
                            stop: 0 #13111C,   
                            stop: 0.5 #3A3042, 
                            stop: 1 #13111C   
                        );
                    }
                """)

        container = QWidget()
        self.setCentralWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.addStretch()

        #Camera hardware
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        #Text above
        self.result_label = QLabel("Ready")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_label.setStyleSheet("""
            color: white;
            font-size: 28px;
            font-weight: bold;
        """)

        main_layout.addWidget(
            self.result_label,
            alignment=Qt.AlignmentFlag.AlignCenter
        )

        #Camera frame
        self.camera_label = QLabel("Starting camera...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid #5c5d84; background-color: #1e1e2f;")
        main_layout.addWidget(self.camera_label, alignment = Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setFixedSize(640, 360)
        #self.camera_label.setFixedSize(704, 396)

        #Buttons
        self.capture_button = QPushButton("Capture")
        self.reset_button = QPushButton("Reset")

        self.capture_button.setFixedSize(315, 50)
        self.reset_button.setFixedSize(315, 50)

        self.capture_button.clicked.connect(self.capture_clicked)
        self.reset_button.clicked.connect(self.reset_clicked)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.capture_button)
        button_layout.addSpacing(0)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        button_style = """
                QPushButton {
                    background-color: #3A3042;
                    color: #e2defa;
                    border: 1px solid #5c4d69;
                    border-radius: 14px;
                    font-size: 16px;
                    font-weight: bold;
                }

                QPushButton:hover {
                    background-color: #4A3E56;
                    border: 1px solid #00c2c7; /* Subtle mint edge glow on hover */
                    color: white;
                }

                QPushButton:pressed {
                    background-color: #231c28;
                    border: 1px solid #3A3042;
                }
                """

        self.capture_button.setStyleSheet(button_style)
        self.reset_button.setStyleSheet(button_style)
        main_layout.addStretch()
        main_layout.setSpacing(20)


    #Methods
    def capture_clicked(self):

        if not hasattr(self, "current_frame"):
            return

        if self.model is None:
            self.result_label.setText("Model not loaded")
            return

        self.result_label.setText("Analyzing...")

        input_image = self.prepare_digit_frame(self.current_frame)

        if input_image is None:
            self.result_label.setText("No digit found")
            return

        # Predict
        prediction = self.model.predict(
            input_image,
            verbose=0
        )

        print(prediction)

        digit = prediction.argmax()
        confidence = prediction[0][digit] * 100

        self.result_label.setText(
            f"Detected Number: {digit}"
        )

    def prepare_digit_frame(self, frame):
        size = 28

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        pixels = gray.astype("uint8")

        cv2.imwrite("debug_gray.png", pixels)

        corners = [
            pixels[0, 0],
            pixels[0, -1],
            pixels[-1, 0],
            pixels[-1, -1]
        ]
        background = np.median(corners)

        if background > 127:
            digit_mask = pixels < background - 80
        else:
            digit_mask = pixels > background + 40

        mask = digit_mask.astype("uint8") * 255
        cv2.imwrite("debug_thresh.png", mask)

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if component_count <= 1:
            return None

        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component = component_areas.argmax() + 1
        if stats[largest_component, cv2.CC_STAT_AREA] < 30:
            return None

        digit = labels == largest_component
        rows, cols = np.where(digit)
        if len(rows) == 0:
            return None

        top = rows.min()
        bottom = rows.max() + 1
        left = cols.min()
        right = cols.max() + 1

        digit = digit[top:bottom, left:right].astype("uint8") * 255
        cv2.imwrite("debug_crop.png", digit)

        crop_height, crop_width = digit.shape
        scale = 20 / max(crop_height, crop_width)
        new_width = max(1, int(crop_width * scale))
        new_height = max(1, int(crop_height * scale))
        digit = cv2.resize(digit, (new_width, new_height), interpolation=cv2.INTER_AREA)

        final_image = np.zeros((size, size), dtype="uint8")
        x = (size - new_width) // 2
        y = (size - new_height) // 2
        final_image[y:y + new_height, x:x + new_width] = digit

        cv2.imwrite("debug_input.png", final_image)

        final_pixels = final_image.astype("float32") / 255.0
        return final_pixels.reshape(1, size, size, 1)

    def reset_clicked(self):
        print("Resetting the system...")
        self.result_label.setText("Ready")

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.current_frame = frame

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.camera_label.width(),
                self.camera_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            #Display
            self.camera_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        #Release Mac camera when user exits the window
        self.camera.release()

app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()
