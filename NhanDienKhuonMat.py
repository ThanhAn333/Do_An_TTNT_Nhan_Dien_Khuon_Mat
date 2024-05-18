import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

class FaceDetectionApp:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_eye.xml')
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise IOError('Không thể tải tệp xml của bộ phân loại cascade khuôn mặt hoặc mắt')

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        return image, len(faces)

    def detect_eyes(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_count = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                eyes_count += 1
        return image, eyes_count

def main():
    st.title("Nhận dạng khuôn mặt và mắt")

    app = FaceDetectionApp()

    option = st.radio("Chọn chức năng:", ("Nhận dạng khuôn mặt", "Nhận dạng mắt", "Tải video và ghi lại", "Nhận dạng từ webcam và ghi lại video"))

    if option == "Nhận dạng khuôn mặt" or option == "Nhận dạng mắt":
        uploaded_file = st.file_uploader("Tải ảnh lên", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            st.image(image, channels="BGR", caption="Ảnh gốc")

            if option == "Nhận dạng khuôn mặt":
                detected_image, num_faces = app.detect_faces(image.copy())
                st.image(detected_image, channels="BGR", caption=f"Ảnh sau khi nhận dạng khuôn mặt. Số lượng khuôn mặt: {num_faces}")
            else:
                detected_image, num_eyes = app.detect_eyes(image.copy())
                st.image(detected_image, channels="BGR", caption=f"Ảnh sau khi nhận dạng mắt. Số lượng mắt: {num_eyes}")

    elif option == "Tải video và ghi lại":
        uploaded_video = st.file_uploader("Tải video lên", type=['mp4'])
        if uploaded_video is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, uploaded_video.name)
                with open(video_path, 'wb') as f:
                    f.write(uploaded_video.read())

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Không thể mở video. Vui lòng thử lại.")
                    return

                output_video_path = "D:\\Python\\C\\Đồ-Án-TTNT\\NhanDienKhuonMat\\Code_Tren_Web\\Do_An_TTNT_Nhan_Dien_Khuon_Mat\\output1.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    detected_frame, _ = app.detect_faces(frame)
                    detected_frame, _ = app.detect_eyes(detected_frame)
                    out.write(detected_frame)

                cap.release()
                out.release()

                st.success(f"Video đã được xử lý và lưu lại tại {output_video_path}")

    elif option == "Nhận dạng từ webcam và ghi lại video":
        st.write("Đang mở webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không thể mở webcam. Vui lòng kiểm tra kết nối và thử lại.")
            return

        output_video_path = None
        recording = False
        detected = False  # Biến để theo dõi việc nhận dạng khuôn mặt và mắt
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể đọc khung hình từ webcam. Vui lòng thử lại.")
                break

            detected_frame, _ = app.detect_faces(frame)
            detected_frame, _ = app.detect_eyes(detected_frame)
            st.image(detected_frame, channels="BGR", caption="Nhận dạng khuôn mặt và mắt từ webcam")
        # Kiểm tra xem đã nhận dạng được khuôn mặt và mắt chưa
            if not detected:
            # Thực hiện nhận dạng khuôn mặt và mắt
                _, num_faces = app.detect_faces(frame)
                _, num_eyes = app.detect_eyes(frame)
            
            # Nếu nhận dạng được khuôn mặt và mắt, đặt biến detected thành True
                if num_faces > 0 and num_eyes > 0:
                    detected = True
                # Dừng ghi video và kết thúc vòng lặp
                    break

if __name__ == "__main__":
    main()
