from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import torch
import cv2
import numpy as np
from uuid import uuid4
from scipy.spatial.transform import Rotation as R

# Add DECA root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOADS_DIR = os.path.join(ROOT_DIR, 'web', 'uploads')
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'web', 'outputs')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'web', 'frontend')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Initialize DECA once
use_cpu_only = os.environ.get('USE_CPU_ONLY', '0') == '1'
if use_cpu_only:
    device = 'cpu'
    print("--- FORCING CPU MODE via environment variable ---")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"--- DECA Initializing on device: {device.upper()} ---")
if device == 'cuda':
    print(f"--- GPU Name: {torch.cuda.get_device_name(0)} ---")

deca_cfg.model.use_tex = False 
deca_cfg.model.extract_tex = True
deca_cfg.rasterizer_type = 'standard'
deca = DECA(config=deca_cfg, device=device)

# Initialize Detector
from facenet_pytorch import MTCNN
detector = MTCNN(keep_all=True, device=device)

def crop_face(image, bbox, scale=1.25):
    """Crop face from image based on bbox [left, top, right, bottom]"""
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    size = int(old_size * scale)
    
    # Calculate crop boundaries
    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2], 
        [center[0] - size / 2, center[1] + size / 2], 
        [center[0] + size / 2, center[1] - size / 2]
    ]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 223], [223, 0]]).astype(np.float32)
    
    tform = cv2.getAffineTransform(src_pts, dst_pts)
    dst_image = cv2.warpAffine(image, tform, (224, 224))
    return dst_image, tform

@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    try:
        request_id = str(uuid4())
        input_filename = f"{request_id}_{file.filename}"
        input_path = os.path.join(UPLOADS_DIR, input_filename)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_ext = file.filename.split('.')[-1].lower()
        is_video = file_ext in ['mp4', 'avi', 'mov', 'mkv']
        
        if is_video:
            # Video Processing
            cap = cv2.VideoCapture(input_path)
            frames = []
            count = 0
            max_frames = 100 # Increased for longer sequences
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret: break
                if count % 5 == 0: # Sample every 5th frame
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                count += 1
            cap.release()
            
            if not frames:
                raise HTTPException(status_code=400, detail="Could not extract any frames from video")

            sequence_results = []
            video_tex_url = None
            
            for f_idx, frame_rgb in enumerate(frames):
                # Detect and process ONLY the main face for video sequence
                bboxes, _ = detector.detect(frame_rgb)
                if bboxes is None or len(bboxes) == 0: continue
                
                # Pick largest bbox
                bbox = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[0]
                face_img, _ = crop_face(frame_rgb, bbox)
                
                input_tensor = torch.from_numpy(face_img.astype(np.float32) / 255.0).permute(2, 0, 1)[None, ...].to(device)
                with torch.no_grad():
                    codedict = deca.encode(input_tensor)
                    
                    # Fix head rotation (Pose contains [rotation, jaw])
                    # Zero out the first 3 parameters (global rotation) 
                    # to keep the head fixed while allowing mouth/expression movement
                    codedict['pose'][:, :3] = 0.0
                    
                    opdict, _ = deca.decode(codedict)
                
                # Extract stats for video frame
                pose = codedict['pose'][0].cpu().numpy()
                exp = codedict['exp'][0].cpu().numpy()
                r = R.from_rotvec(pose[:3])
                euler = r.as_euler('xyz', degrees=True)
                mouth_open = R.from_rotvec(pose[3:]).as_euler('xyz', degrees=True)[0]
                
                face_status = {
                    "pose": {"pitch": round(float(euler[0]), 2), "yaw": round(float(euler[1]), 2), "roll": round(float(euler[2]), 2)},
                    "expression": {"intensity": round(float(np.linalg.norm(exp)), 2), "mouth_open": round(float(mouth_open), 2)}
                }

                output_subdir = os.path.join(OUTPUTS_DIR, f"{request_id}_f{f_idx}")
                os.makedirs(output_subdir, exist_ok=True)
                obj_name = "face.obj"
                deca.save_obj(os.path.join(output_subdir, obj_name), opdict)
                
                # Extract texture for the FIRST frame only to save time
                if video_tex_url is None:
                    tex_name = "face.png"
                    if os.path.exists(os.path.join(output_subdir, tex_name)):
                        video_tex_url = f"/outputs/{request_id}_f{f_idx}/{tex_name}"
                
                sequence_results.append({
                    "frame": f_idx,
                    "obj_url": f"/outputs/{request_id}_f{f_idx}/{obj_name}",
                    "face_status": face_status
                })
            
            return {
                "id": request_id,
                "type": "video",
                "sequence": sequence_results,
                "tex_url": video_tex_url
            }

        else:
            # Image Processing (Existing Logic)
            full_image = cv2.imread(input_path)
            if full_image is None:
                raise HTTPException(status_code=400, detail="Could not read uploaded image. Unsupported format?")
                
            full_image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            
            bboxes, probs = detector.detect(full_image_rgb)
            if bboxes is None or len(bboxes) == 0:
                bboxes = [[0, 0, full_image.shape[1], full_image.shape[0]]]
            
            results = []
            for i, bbox in enumerate(bboxes):
                face_img, tform = crop_face(full_image_rgb, bbox)
                input_tensor = torch.from_numpy(face_img.astype(np.float32) / 255.0).permute(2, 0, 1)[None, ...].to(device)
                
                with torch.no_grad():
                    codedict = deca.encode(input_tensor)
                    opdict, visdict = deca.decode(codedict)
                    
                pose = codedict['pose'][0].cpu().numpy()
                exp = codedict['exp'][0].cpu().numpy()
                
                r = R.from_rotvec(pose[:3])
                euler = r.as_euler('xyz', degrees=True)
                mouth_open = R.from_rotvec(pose[3:]).as_euler('xyz', degrees=True)[0]
                
                face_status = {
                    "pose": {"pitch": round(float(euler[0]), 2), "yaw": round(float(euler[1]), 2), "roll": round(float(euler[2]), 2)},
                    "expression": {"intensity": round(float(np.linalg.norm(exp)), 2), "mouth_open": round(float(mouth_open), 2)}
                }

                output_subdir = os.path.join(OUTPUTS_DIR, f"{request_id}_{i}")
                os.makedirs(output_subdir, exist_ok=True)
                obj_name = "face.obj"
                deca.save_obj(os.path.join(output_subdir, obj_name), opdict)
                tex_name = "face.png"
                has_tex = os.path.exists(os.path.join(output_subdir, tex_name))
                
                results.append({
                    "index": i,
                    "obj_url": f"/outputs/{request_id}_{i}/{obj_name}",
                    "tex_url": f"/outputs/{request_id}_{i}/{tex_name}" if has_tex else None,
                    "face_status": face_status
                })
                
            return {
                "id": request_id,
                "type": "image",
                "faces": results
            }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Static files
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
