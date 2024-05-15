import cv2
import os
import streamlit as st

# Tải tệp Haar cascade cho khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceDetectionApp:
    def __init__(self):
        # Tạo thư mục để lưu ảnh khuôn mặt được phát hiện, ảnh chụp, và video
        self.output_dir = 'detected_faces'
        self.snapshot_dir = 'snapshots'
        self.video_output_dir = 'recorded_videos'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)

        # Khởi tạo các biến
        self.current_camera = 0
        self.face_counter = 0
        self.snapshot_counter = 0
        self.recording = False
        self.video_writer = None
        self.zoom_factor = 1.0
        self.zoom_step = 0.1

    def start_video_capture(self):
        # Stream dữ liệu video từ camera
        st.header("Video Stream")
        cap = cv2.VideoCapture(self.current_camera)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error('Không thể nhận dữ liệu từ camera.')
                break

            # Điều chỉnh mức phóng đại của camera
            frame = self._zoom_frame(frame)

            # Phát hiện khuôn mặt và lưu vào thư mục output_dir
            frame, num_faces = self._detect_faces(frame)

            # Hiển thị frame trong Streamlit
            st.image(frame, channels="BGR", use_column_width=True)

            # Ghi video nếu đang ghi và lưu vào thư mục video_output_dir
            self._record_video(frame)

            # Kiểm tra nếu cửa sổ bị đóng
            if cv2.getWindowProperty('Trinh phat hien khuon mat', cv2.WND_PROP_VISIBLE) < 1:
                self.stop_video_capture(cap)
                break

            # Đợi 30ms trước khi lặp lại
            cv2.waitKey(30)

        # Kết thúc video capture khi kết thúc vòng lặp
        cap.release()
        self._release_video_writer()

    def _zoom_frame(self, frame):
        # Điều chỉnh phóng đại của frame
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        new_width, new_height = int(width / self.zoom_factor), int(height / self.zoom_factor)
        x1, y1 = max(0, center_x - new_width // 2), max(0, center_y - new_height // 2)
        x2, y2 = min(width, center_x + new_width // 2), min(height, center_y + new_height // 2)
        return cv2.resize(frame[y1:y2, x1:x2], (width, height))

    def _detect_faces(self, frame):
        # Chuyển đổi sang ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Chạy bộ phát hiện khuôn mặt trên ảnh xám
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Vẽ hình chữ nhật xung quanh các khuôn mặt
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
            # Trích xuất và lưu khuôn mặt vào thư mục output_dir
            face = frame[y:y+h, x:x+w]
            face_filename = os.path.join(self.output_dir, f'face_{self.face_counter}.png')
            cv2.imwrite(face_filename, face)
            self.face_counter += 1

        # Thêm số lượng khuôn mặt được phát hiện vào frame
        num_faces = len(face_rects)
        text = f'So luong khuon mat phat hien: {num_faces}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame, num_faces

    def _record_video(self, frame):
        # Ghi video nếu đang ghi và lưu vào thư mục video_output_dir
        if self.recording:
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_filename = os.path.join(self.video_output_dir, f'video_{self.snapshot_counter}.avi')
                self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            self.video_writer.write(frame)
            self.snapshot_counter += 1

    def _release_video_writer(self):
        # Giải phóng video writer khi kết thúc
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None

    def stop_video_capture(self, cap):
        # Dừng video capture và giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()

    def toggle_recording(self):
        # Bật/tắt ghi video
        self.recording = not self.recording

    def switch_camera(self):
        # Chuyển đổi giữa camera trước và sau
        self.current_camera = 1 - self.current_camera

    def zoom_in(self):
        # Phóng to hình ảnh
        self.zoom_factor = min(self.zoom_factor + self.zoom_step, 3.0)

    def zoom_out(self):
        # Thu nhỏ hình ảnh
        self.zoom_factor = max(self.zoom_factor - self.zoom_step, 1.0)


# Tạo đối tượng ứng dụng và chạy ứng dụng
app = FaceDetectionApp()

# Hiển thị các nút điều khiển trong Streamlit
st.sidebar.header("Controls")
if st.sidebar.button("Start"):
    app.start_video_capture()
if st.sidebar.button("Toggle Recording"):
    app.toggle_recording()
if st.sidebar.button("Switch Camera"):
    app.switch_camera()
if st.sidebar.button("Zoom In"):
    app.zoom_in()
if st.sidebar.button("Zoom Out"):
    app.zoom_out()