import cv2
import os
import time
import streamlit as st

# Tải tệp Haar cascade cho khuôn mặt
cascade_path = 'haar_cascade_files/haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    st.error(f"Không tìm thấy tệp cascade tại đường dẫn: {cascade_path}")
else:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error(f"Không thể tải tệp cascade từ: {cascade_path}")

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
        self.cap = None
        self.zoom_factor = 1.0
        self.zoom_step = 0.1

    def start_video_capture(self):
        # Mở camera
        self.cap = cv2.VideoCapture(self.current_camera)

        # Kiểm tra xem camera có được mở thành công hay không
        if not self.cap.isOpened():
            st.error('Không thể mở camera. Vui lòng kiểm tra quyền truy cập và kết nối của camera.')
            return

        st.success("Camera đã được mở thành công.")
        
        frame_placeholder = st.empty()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                st.error('Không thể nhận dữ liệu từ camera.')
                break

            # Điều chỉnh mức phóng đại của camera
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            new_width, new_height = int(width / self.zoom_factor), int(height / self.zoom_factor)
            x1, y1 = max(0, center_x - new_width // 2), max(0, center_y - new_height // 2)
            x2, y2 = min(width, center_x + new_width // 2), min(height, center_y + new_height // 2)
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (width, height))

            # Chuyển đổi sang ảnh xám
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Chạy bộ phát hiện khuôn mặt trên ảnh xám
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Vẽ hình chữ nhật xung quanh các khuôn mặt và lưu khuôn mặt vào đĩa
            for (x, y, w, h) in face_rects:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Trích xuất khuôn mặt
                face = frame[y:y+h, x:x+w]

                # Lưu khuôn mặt vào đĩa
                face_filename = os.path.join(self.output_dir, f'face_{self.face_counter}.png')
                cv2.imwrite(face_filename, face)
                self.face_counter += 1
            # Hiển thị số lượng khuôn mặt được phát hiện
            cv2.putText(frame, f'Số lượng khuôn mặt: {len(face_rects)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            # Hiển thị frame trong Streamlit
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

            # Ghi video nếu đang ghi
            if self.recording:
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_filename = os.path.join(self.video_output_dir, f'video_{self.snapshot_counter}.avi')
                    self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                self.video_writer.write(frame)

            time.sleep(0.03)  # Đợi 30ms trước khi xử lý frame tiếp theo

    def toggle_recording(self):
        # Ghi video khi nút được nhấn
        self.recording = not self.recording

    def switch_camera(self):
        # Chuyển đổi giữa camera trước và sau
        self.current_camera = 1 - self.current_camera
        if self.cap is not None:
            self.cap.release()
        self.start_video_capture()

    def zoom_in(self):
        # Phóng to hình ảnh
        self.zoom_factor = min(self.zoom_factor + self.zoom_step, 3.0)

    def zoom_out(self):
        # Thu nhỏ hình ảnh
        self.zoom_factor = max(self.zoom_factor - self.zoom_step, 1.0)

    def stop_video_capture(self):
        # Dừng video capture và giải phóng tài nguyên
        if self.cap is not None:
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

# Tạo đối tượng ứng dụng và chạy ứng dụng
app = FaceDetectionApp()

# Hiển thị các nút điều khiển trong Streamlit
# Hiển thị các nút điều khiển trong Streamlit
st.sidebar.markdown("### Điều khiển")
if st.sidebar.button("Bắt đầu"):
    app.start_video_capture()
if st.sidebar.button("Bật/Tắt ghi hình"):
    app.toggle_recording()
if st.sidebar.button("Chuyển camera"):
    app.switch_camera()
if st.sidebar.button("Phóng to"):
    app.zoom_in()
if st.sidebar.button("Thu nhỏ"):
    app.zoom_out()
if st.sidebar.button("Dừng"):
    app.stop_video_capture()