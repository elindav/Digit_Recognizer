import sys
import cv2
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
            self.model = load_model("model.h5")
            print("AI model loaded successfully")
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

        frame = self.current_frame

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray,
            150,
            255,
            cv2.THRESH_BINARY_INV
        )

        resized = cv2.resize(thresh, (28, 28))

        # Save debug image
        cv2.imwrite("debug_input.png", resized)

        # Normalize
        resized = resized.astype("float32") / 255.0

        # Prepare input shape
        input_image = resized.reshape(1, 28, 28)

        # Predict
        prediction = self.model.predict(
            input_image,
            verbose=0
        )

        print(prediction)

        digit = prediction.argmax()

        self.result_label.setText(
            f"Detected Number: {digit}"
        )

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