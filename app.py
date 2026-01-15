import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# إعداد الصفحة
st.set_page_config(page_title="مدرب المقابلة المباشر", page_icon="🔴")

st.title("🔴 Live AI Interview Coach")
st.markdown("### المدرب الشخصي المباشر: انظر للكاميرا وسأخبرك بتركيزك فوراً!")

# إعدادات الشريط الجانبي
sensitivity = st.sidebar.slider("حساسية الحركة", 0.5, 2.0, 1.0)

# تجهيز MediaPipe مرة واحدة
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # تحويل الصورة من تنسيق الكاميرا إلى OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # تجهيز الصورة
        img_h, img_w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(img_rgb)

        status = "Looking Away ⚠️"
        color = (0, 0, 255) # أحمر
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # النقاط: الأنف (1)، الأذن اليسرى (234)، الأذن اليمنى (454)
                nose = face_landmarks.landmark[1]
                left_ear = face_landmarks.landmark[234]
                right_ear = face_landmarks.landmark[454]
                
                nx, ny = int(nose.x * img_w), int(nose.y * img_h)
                lx = int(left_ear.x * img_w)
                rx = int(right_ear.x * img_w)

                # حساب المسافات لتحديد الاتجاه
                dist_left = abs(nx - lx)
                dist_right = abs(nx - rx)
                
                try:
                    ratio = dist_left / dist_right
                    
                    # معادلة التركيز (قابلة للتعديل بالحساسية)
                    if (0.5 / sensitivity) < ratio < (2.0 * sensitivity):
                        status = "Focused ✅"
                        color = (0, 255, 0) # أخضر
                    
                    # رسم دائرة على الأنف
                    cv2.circle(img, (nx, ny), 5, color, -1)
                    
                except:
                    pass

        # كتابة الحالة على الشاشة
        cv2.putText(img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # إرجاع الصورة المعالجة
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# تشغيل الكاميرا المباشرة
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

st.info("💡 ملاحظة: إذا كنت تستخدم الهاتف، قد يطلب منك المتصفح الإذن للكاميرا. تأكد من إعطاء الصلاحية.")