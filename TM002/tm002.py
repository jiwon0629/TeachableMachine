import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, 500, 400)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 400, 200)
        self.image_label.setScaledContents(True)

        self.class_label = QLabel(self)
        self.class_label.setGeometry(50, 270, 400, 30)

        self.confidence_label = QLabel(self)
        self.confidence_label.setGeometry(50, 300, 400, 30)

        self.select_button = QPushButton("Select Image", self)
        self.select_button.setGeometry(200, 350, 100, 30)
        self.select_button.clicked.connect(self.select_image)

        self.model = load_model("./model/keras_Model.h5", compile=False)
        self.class_names = open("./model/labels.txt", "r").readlines()

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.png)")
        if file_path:
            image = Image.open(file_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = self.model.predict(data)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]

            self.display_image(file_path)
            self.display_prediction(class_name[2:], confidence_score)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap)

    def display_prediction(self, class_name, confidence_score):
        self.class_label.setText(f"Class: {class_name}")
        self.confidence_label.setText(f"Confidence Score: {confidence_score:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
