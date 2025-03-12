import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        # 윈도우 초기화
        super().__init__()
        self.setWindowTitle("Image Classifier")  # 윈도우 제목 설정
        self.setGeometry(100, 100, 500, 400)  # 윈도우 위치와 크기 설정

        # 이미지 표시용 레이블
        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 400, 200)  # 위치와 크기 설정
        self.image_label.setScaledContents(True)  # 이미지가 레이블 크기에 맞게 축소되도록 설정

        # 클래스 이름을 표시할 레이블
        self.class_label = QLabel(self)
        self.class_label.setGeometry(50, 270, 400, 30)  # 위치와 크기 설정

        # 신뢰도를 표시할 레이블
        self.confidence_label = QLabel(self)
        self.confidence_label.setGeometry(50, 300, 400, 30)  # 위치와 크기 설정

        # 이미지 선택 버튼
        self.select_button = QPushButton("Select Image", self)
        self.select_button.setGeometry(200, 350, 100, 30)  # 버튼의 위치와 크기 설정
        self.select_button.clicked.connect(self.select_image)  # 버튼 클릭 시 이미지 선택 함수 호출

        # 사전 훈련된 모델 로드 (파일 경로는 './model/keras_Model.h5'로 가정)
        self.model = load_model("./model/keras_Model.h5", compile=False)

        # 클래스 이름 목록 로드 (파일 경로는 './model/labels.txt'로 가정)
        self.class_names = open("./model/labels.txt", "r").readlines()

    def select_image(self):
        # 이미지 파일 선택 대화상자 열기
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.png)")
        if file_path:
            # 선택한 이미지 파일 열기
            image = Image.open(file_path).convert("RGB")
            
            # 이미지 크기를 모델 입력 크기에 맞게 리사이징 (224x224)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            
            # 이미지를 배열로 변환하고 정규화
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            
            # 모델에 맞는 입력 형태로 데이터 배열 준비
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # 모델을 사용하여 예측 수행
            prediction = self.model.predict(data)
            
            # 예측된 클래스 인덱스 찾기
            index = np.argmax(prediction)
            
            # 예측된 클래스 이름과 신뢰도 가져오기
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]

            # 이미지와 예측 결과를 화면에 표시
            self.display_image(file_path)
            self.display_prediction(class_name[2:], confidence_score)

    def display_image(self, file_path):
        # 선택된 이미지를 화면에 표시
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap)

    def display_prediction(self, class_name, confidence_score):
        # 예측된 클래스 이름과 신뢰도 표시
        self.class_label.setText(f"Class: {class_name}")
        self.confidence_label.setText(f"Confidence Score: {confidence_score:.2f}")

# 메인 프로그램 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication 객체 생성
    window = MainWindow()  # MainWindow 객체 생성
    window.show()  # 윈도우를 화면에 표시
    sys.exit(app.exec_())  # 애플리케이션 실행
