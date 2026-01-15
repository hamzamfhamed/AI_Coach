import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# إعداد الصفحة
st.set_page_config(page_title="مدرب المقابلة المباشر", page_icon="🔴")

st.title("🔴 Live AI Interview Coach")
st.markdown("### المدرب الشخصي المباشر: انظر للكاميرا وسأخبرك بتركيزك فوراً!")

# إعدادات الشريط الجانبي
sensitivity = st.sidebar.slider("حساسية الحركة", 0.5, 2.0, 1.0)

# تجهيز MediaPipe (تقليل الدقة لزيادة السرعة)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=False,  # ألغينا النقاط الدقيقة لتسريع المعالجة
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# إعدادات السيرفر لتقليل التقطيع
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # تحويل الصورة
        img = frame.to_ndarray(format="bgr24")
        
        # تصغير الصورة قليلاً للمعالجة السريعة (اختياري)
        # img = cv2.resize(img, (640, 480))
        
        img_h, img_w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # المعالجة
        results = face_mesh.process(img_rgb)

        status = "Looking Away ⚠️"
        color = (0, 0, 255) # أحمر
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                left_ear = face_landmarks.landmark[234]
                right_ear = face_landmarks.landmark[454]
                
                nx, ny = int(nose.x * img_w), int(nose.y * img_h)
                lx = int(left_ear.x * img_w)
                rx = int(right_ear.x * img_w)

                dist_left = abs(nx - lx)
                dist_right = abs(nx - rx)
                
                try:
                    ratio = dist_left / dist_right
                    if (0.5 / sensitivity) < ratio < (2.0 * sensitivity):
                        status = "Focused ✅"
                        color = (0, 255, 0)
                    
                    cv2.circle(img, (nx, ny), 5, color, -1)
                except:
                    pass

        cv2.putText(img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# تشغيل الكاميرا بإعدادات مخففة
webrtc_streamer(
    key="ai-coach",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"width": 480, "height": 360, "frameRate": 15}, # سرعة بدلاً من جودة
        "audio": False
    }
)