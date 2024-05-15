import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox,Scale

class FaceDetectionApp:
    def __init__(self):
        # Tải tệp Haar cascade cho khuôn mặt
        self.face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')

        # Kiểm tra xem các tệp cascade đã được tải đúng cách chưa
        if self.face_cascade.empty():
            raise IOError('Không thể tải tệp xml của bộ phân loại cascade khuôn mặt')

        # Tạo thư mục để lưu ảnh khuôn mặt được phát hiện, ảnh chụp, và video
        self.output_dir = 'detected_faces'
        self.snapshot_dir = 'snapshots'
        self.video_output_dir = 'recorded_videos'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        if not os.path.exists(self.video_output_dir):
            os.makedirs(self.video_output_dir)

        # Khởi tạo các biến
        self.current_camera = 0
        self.face_counter = 0
        self.snapshot_counter = 0
        self.recording = False
        self.video_writer = None
        self.start_time = None
        self.zoom_factor = 1.0
        self.zoom_step = 0.1

        # Tạo giao diện người dùng
        self.root = tk.Tk()
        self.root.title("Trinh phat hien khuon mat")
        self.root.geometry("400x300")

        # Tạo các nút điều khiển
        self.start_button = ttk.Button(self.root, text="Bắt đầu", command=self.start_video_capture)
        self.start_button.pack(pady=5)

        self.stop_button = ttk.Button(self.root, text="Dừng", command=self.stop_video_capture)
        self.stop_button.pack(pady=5)

        self.snapshot_button = ttk.Button(self.root, text="Chụp ảnh", command=self.capture_snapshot)
        self.snapshot_button.pack(pady=5)

        self.record_button = ttk.Button(self.root, text="Ghi video", command=self.toggle_recording)
        self.record_button.pack(pady=5)

        self.switch_button = ttk.Button(self.root, text="Chuyển camera", command=self.switch_camera)
        self.switch_button.pack(pady=5)

        self.zoom_in_button = ttk.Button(self.root, text="Zoom in", command=self.zoom_in)
        self.zoom_in_button.pack(pady=5)

        self.zoom_out_button = ttk.Button(self.root, text="Zoom out", command=self.zoom_out)
        self.zoom_out_button.pack(pady=5)

        self.zoom_scale = Scale(self.root, from_=1.0, to=3.0, resolution=0.1, orient='horizontal', label='Zoom Factor', command=self.update_zoom_factor)
        self.zoom_scale.pack(pady=5)

    def start_video_capture(self):
        self.cap = cv2.VideoCapture(self.current_camera)
        self.process_video()

    def process_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

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
        face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Vẽ hình chữ nhật xung quanh các khuôn mặt
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Trích xuất khuôn mặt
            face = frame[y:y+h, x:x+w]

            # Lưu khuôn mặt vào đĩa
            face_filename = os.path.join(self.output_dir, f'face_{self.face_counter}.png')
            cv2.imwrite(face_filename, face)
            self.face_counter += 1

        # Thêm lớp phủ văn bản với số lượng khuôn mặt được phát hiện
        num_faces = len(face_rects)
        text = f'So luong khuon mat phat hien: {num_faces}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Hiển thị bộ đếm thời gian nếu đang ghi video
        if self.recording:
            elapsed_time = time.time() - self.start_time
            timer_text = f'Time: {int(elapsed_time)}s'
            cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Hiển thị đầu ra
        cv2.imshow('Trinh phat hien khuon mat', frame)

        # Ghi video nếu đang ghi
        if self.recording:
            self.video_writer.write(frame)

        # Kiểm tra nếu cửa sổ bị đóng
        if cv2.getWindowProperty('Trinh phat hien khuon mat', cv2.WND_PROP_VISIBLE) < 1:
            self.stop_video_capture()
            return

        self.root.after(10, self.process_video)

    def stop_video_capture(self):
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def capture_snapshot(self):
        snapshot_filename = os.path.join(self.snapshot_dir, f'snapshot_{self.snapshot_counter}.png')
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(snapshot_filename, frame)
            self.snapshot_counter += 1
            messagebox.showinfo("Thông báo", f'Ảnh đã được chụp và lưu vào {snapshot_filename}')

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            video_filename = os.path.join(self.video_output_dir, f'video_{self.snapshot_counter}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            ret, frame = self.cap.read()
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            messagebox.showinfo("Thông báo", f'Bắt đầu ghi video: {video_filename}')
        else:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            messagebox.showinfo("Thông báo", 'Dừng ghi video')

    def switch_camera(self):
        self.current_camera = 1 - self.current_camera
        self.cap.release()
        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", f'Không thể mở camera {self.current_camera}')
            self.current_camera = 1 - self.current_camera  # Khôi phục giá trị camera về trạng thái trước
        else:
            messagebox.showinfo("Thông báo", f'Đã chuyển sang camera {self.current_camera}')

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + self.zoom_step, 3.0)

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - self.zoom_step, 1.0)

    def run(self):
        self.root.mainloop()
    
    def update_zoom_factor(self, value):
        self.zoom_factor = float(value)


    def detect_faces(self, frame):
    # Phát hiện khuôn mặt trong frame
        face_rects = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
        for (x, y, w, h) in face_rects:
        # Vẽ hình chữ nhật xung quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            

        return frame
# Khởi chạy ứng dụng
app = FaceDetectionApp()
app.run()
