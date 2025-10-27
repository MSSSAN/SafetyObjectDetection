import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO, RTDETR
import tempfile
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import torch

# Detect and set device
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, "CUDA"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, "MPS"
    else:
        device = torch.device("cpu")
        return device, "CPU"

# Define colors for each class (BGR format)
CLASS_COLORS = {
    'Person': (255, 255, 255),      # White
    'gloves': (0, 128, 0),          # Green
    'goggles': (0, 128, 0),         # Green
    'helmet': (0, 128, 0),          # Green
    'no_gloves': (0, 0, 255),       # Red (BGR format)
    'no_goggle': (0, 0, 255),       # Red (BGR format)
    'no_helmet': (0, 0, 255)        # Red (BGR format)
}

# Load models based on selection
@st.cache_resource
def load_yolo_model(model_path):
    device, device_name = get_device()
    model = YOLO(model_path)
    model.to(device)
    print(f"YOLO model loaded on device: {device}")
    return model, device, device_name

@st.cache_resource
def load_rtdetr_model(model_path):
    device, device_name = get_device()
    model = RTDETR(model_path)
    model.to(device)
    print(f"RT-DETR model loaded on device: {device}")
    return model, device, device_name

st.title("개인보호장비(PPE) 감지 시스템")
st.write("적절한 개인 보호 장비의 착용 여부를 감지합니다.")

# Model selection in sidebar (outside of modes)
st.sidebar.header("모델 선택")
model_type = st.sidebar.radio(
    "감지 모델 선택:",
    ["YOLOv11", "RT-DETR v1", "YOLOv11 vs RT-DETR v1"],
)

# Model paths

yolo_model_path = '/Users/abc/Downloads/yolo-v11-s_E50_BS32_Baseline_mAP_0.777.pt'  # Replace with your YOLO model path
rtdetr_model_path = '/Users/abc/Downloads/best_rtdetr_x_oct23_1019am.pt'  # Replace with your RT-DETR model path

# Load selected model(s)
if model_type == "YOLOv11":
    model, device, device_name = load_yolo_model(yolo_model_path)
    st.sidebar.info(f"📦 로드됨: YOLOv11")
    comparison_mode = False
elif model_type == "RT-DETR v1":
    model, device, device_name = load_rtdetr_model(rtdetr_model_path)
    st.sidebar.info(f"📦 로드됨: RT-DETR v1")
    comparison_mode = False
else:  # Comparison mode
    yolo_model, yolo_device, yolo_device_name = load_yolo_model(yolo_model_path)
    rtdetr_model, rtdetr_device, rtdetr_device_name = load_rtdetr_model(rtdetr_model_path)
    model = yolo_model  # Default for non-image modes
    device = yolo_device
    device_name = yolo_device_name
    st.sidebar.info(f"📦 로드됨: 두 모델 모두")
    comparison_mode = True

# Display device info in small grey text
st.sidebar.markdown(f"<p style='font-size:11px; color:#666666;'>{device_name} 사용 중</p>", unsafe_allow_html=True)

st.sidebar.markdown("---")

# PPE Selection
st.sidebar.header("PPE 항목 선택")
st.sidebar.write("감지할 PPE 항목을 선택하세요:")

detect_helmet = st.sidebar.checkbox("헬멧 (Helmet)", value=True)
detect_goggles = st.sidebar.checkbox("고글 (Goggles)", value=True)
detect_gloves = st.sidebar.checkbox("장갑 (Gloves)", value=True)

# Create a set of classes to detect based on selections
classes_to_detect = {'Person'}  # Always detect Person
if detect_helmet:
    classes_to_detect.update(['helmet', 'no_helmet'])
if detect_goggles:
    classes_to_detect.update(['goggles', 'no_goggle'])
if detect_gloves:
    classes_to_detect.update(['gloves', 'no_gloves'])

st.sidebar.markdown("---")

# Function to draw custom colored bounding boxes
def draw_detections(frame, results, classes_to_detect, detection_model, is_bgr=True):
    # For webcam/video: frame is already BGR, for images: frame is RGB
    if is_bgr:
        annotated_frame = frame.copy()
    else:
        # Convert RGB to BGR for processing
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        label = detection_model.names[cls]
        
        # Skip if this class is not selected for detection
        if label not in classes_to_detect:
            continue
        
        # Get color for this class (BGR format)
        color = CLASS_COLORS.get(label, (0, 255, 0))  # Default to green if not found
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f'{label} {conf:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Convert back to RGB only for image display
    if not is_bgr:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame

# Sidebar for mode selection
mode = st.sidebar.selectbox("모드 선택", ["이미지 업로드", "동영상 업로드", "웹캠 (실시간)"])

if mode == "이미지 업로드":
    uploaded_file = st.file_uploader("이미지 업로드", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if comparison_mode:
            # Comparison mode: run both models
            st.subheader("모델 비교")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**YOLOv11 감지**")
                with torch.no_grad():
                    yolo_results = yolo_model(img_array, device=yolo_device)
                yolo_annotated = draw_detections(img_array, yolo_results, classes_to_detect, yolo_model, is_bgr=False)
                st.image(yolo_annotated, caption="YOLOv11 결과", use_container_width=True)
                
                st.write("감지 결과:")
                for box in yolo_results[0].boxes:
                    class_name = yolo_model.names[int(box.cls)]
                    if class_name in classes_to_detect:
                        confidence = float(box.conf)
                        st.write(f"- {class_name} ({confidence:.2f})")
            
            with col2:
                st.write("**RT-DETR v1 감지**")
                with torch.no_grad():
                    rtdetr_results = rtdetr_model(img_array, device=rtdetr_device)
                rtdetr_annotated = draw_detections(img_array, rtdetr_results, classes_to_detect, rtdetr_model, is_bgr=False)
                st.image(rtdetr_annotated, caption="RT-DETR v1 결과", use_container_width=True)
                
                st.write("감지 결과:")
                for box in rtdetr_results[0].boxes:
                    class_name = rtdetr_model.names[int(box.cls)]
                    if class_name in classes_to_detect:
                        confidence = float(box.conf)
                        st.write(f"- {class_name} ({confidence:.2f})")
        
        else:
            # Single model mode
            with torch.no_grad():
                results = model(img_array, device=device)
            
            # Draw custom colored bounding boxes
            annotated_img = draw_detections(img_array, results, classes_to_detect, model, is_bgr=False)
            
            # Display results (image is now in RGB format)
            st.image(annotated_img, caption="감지 결과")
            
            # Show detection details
            st.subheader("감지 결과:")
            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                if class_name in classes_to_detect:
                    confidence = float(box.conf)
                    st.write(f"- {class_name} (신뢰도: {confidence:.2f})")

elif mode == "동영상 업로드":
    uploaded_video = st.file_uploader("동영상 업로드", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        input_path = "input_video.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())
        
        st.success("✅ 동영상이 성공적으로 업로드되었습니다!")
        
        # Processing options
        st.write("**처리 옵션:**")
        process_every_n = st.selectbox(
            "N 프레임마다 처리 (1 = 모든 프레임, 높을수록 빠른 처리):",
            [1, 2, 3, 5, 10],
            index=0
        )
        
        # Process video button
        if st.button("🎬 PPE 감지로 동영상 처리"):
            temp_output = "temp_output.avi"
            output_path = "output_video.mp4"
            
            # Open video
            cap = cv2.VideoCapture(input_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # First save as AVI (more reliable)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            last_detection_result = None
            
            st.info(f"🎥 {device_name}을(를) 사용하여 {fps} FPS로 {total_frames}개 프레임 처리 중...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection on selected frames with explicit device
                if frame_count % process_every_n == 0:
                    with torch.no_grad():
                        results = model(frame, device=device)
                    annotated_frame = draw_detections(frame, results, classes_to_detect, model, is_bgr=True)
                    last_detection_result = annotated_frame
                else:
                    # Use last detection result or original frame
                    annotated_frame = last_detection_result if last_detection_result is not None else frame
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"프레임 {frame_count}/{total_frames} 처리 중 ({progress*100:.1f}%)")
            
            # Release resources
            cap.release()
            out.release()
            
            # Convert to web-compatible MP4 using ffmpeg if available
            status_text.text("웹 호환 형식으로 변환 중...")
            try:
                import subprocess
                subprocess.run([
                    'ffmpeg', '-i', temp_output, '-c:v', 'libx264', 
                    '-preset', 'fast', '-crf', '22', '-y', output_path
                ], check=True, capture_output=True)
                os.remove(temp_output)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg not available, try direct H.264 encoding
                status_text.text("ffmpeg를 찾을 수 없습니다. 대체 방법을 시도합니다...")
                cap2 = cv2.VideoCapture(temp_output)
                fourcc2 = cv2.VideoWriter_fourcc(*'avc1')
                out2 = cv2.VideoWriter(output_path, fourcc2, fps, (width, height))
                
                while cap2.isOpened():
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    out2.write(frame)
                
                cap2.release()
                out2.release()
                os.remove(temp_output)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ 동영상 처리가 완료되었습니다!")
            
            # Store processed video path in session state
            st.session_state.processed_video_path = output_path
        
        # Display processed video
        if 'processed_video_path' in st.session_state and os.path.exists(st.session_state.processed_video_path):
            st.write("---")
            st.write("**PPE 감지가 적용된 처리된 동영상:**")
            
            # Display video with native HTML5 controls (play, pause, seek, speed control)
            st.video(st.session_state.processed_video_path)
            
            st.info("💡 팁: 동영상 컨트롤을 사용하여 재생, 일시정지, 탐색 및 재생 속도를 조정할 수 있습니다. 동영상은 모든 감지 결과가 표시된 상태로 전체 속도로 부드럽게 재생됩니다!")
            
            # Download button
            with open(st.session_state.processed_video_path, "rb") as file:
                st.download_button(
                    label="📥 처리된 동영상 다운로드",
                    data=file,
                    file_name="ppe_detection_output.mp4",
                    mime="video/mp4"
                )

elif mode == "웹캠 (실시간)":
    st.write("WebRTC를 사용한 실시간 웹캠 감지")
    
    if comparison_mode:
        # Comparison mode: display both models side by side
        st.subheader("실시간 모델 비교")
        
        # Single button to start both webcams
        start_comparison = st.checkbox("🎥 두 모델 동시 실행", value=False, key="start_both_webcams")
        
        if start_comparison:
            col1, col2 = st.columns(2)
            
            # Define the video processor class for YOLO
            class YOLODetector(VideoProcessorBase):
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Run detection with explicit device and no_grad for efficiency
                    with torch.no_grad():
                        results = yolo_model(img, device=yolo_device)
                    annotated_img = draw_detections(img, results, classes_to_detect, yolo_model, is_bgr=True)
                    
                    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
            
            # Define the video processor class for RT-DETR
            class RTDETRDetector(VideoProcessorBase):
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Run detection with explicit device and no_grad for efficiency
                    with torch.no_grad():
                        results = rtdetr_model(img, device=rtdetr_device)
                    annotated_img = draw_detections(img, results, classes_to_detect, rtdetr_model, is_bgr=True)
                    
                    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
            
            with col1:
                st.markdown("### YOLOv11")
                # Start YOLO webcam stream with custom styling
                ctx1 = webrtc_streamer(
                    key="yolo-detection",
                    video_processor_factory=YOLODetector,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False}
                )
            
            with col2:
                st.markdown("### RT-DETR v1")
                # Start RT-DETR webcam stream with custom styling
                ctx2 = webrtc_streamer(
                    key="rtdetr-detection",
                    video_processor_factory=RTDETRDetector,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False}
                )
        else:
            st.info("⬆️ 위의 체크박스를 선택하여 두 모델의 실시간 비교를 시작하세요.")
    
    else:
        # Single model mode
        st.info("💡 실시간 웹캠 감지를 시작하려면 아래 버튼을 클릭하세요.")
        
        # Define the video processor class
        class PPEDetector(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # Run detection with explicit device and no_grad for efficiency
                with torch.no_grad():
                    results = model(img, device=device)
                annotated_img = draw_detections(img, results, classes_to_detect, model, is_bgr=True)
                
                return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        
        # Start webcam stream with larger resolution
        webrtc_streamer(
            key="ppe-detection",
            video_processor_factory=PPEDetector,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False}
        )
