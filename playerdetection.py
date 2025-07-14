import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time

class PlayerReidentificationTracker:
    
    def __init__(self, model_path, video_path, output_path):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        
        self.conf_thresh = 0.4
        self.max_distance = 150
        self.max_age = 30
        self.reid_memory_frames = 300
        self.learning_frames = 90
        
        self.feature_dim = 512
        self.appearance_weight = 0.7
        self.motion_weight = 0.3
        self.reid_threshold = 0.6
        
        self.model = YOLO(model_path)
        if hasattr(self.model, 'fuse'):
            self.model.fuse()
        
        print("Available classes in model:")
        for i, name in self.model.names.items():
            print(f"  {i}: {name}")
        
        self.target_class_id = self._find_target_class()
        print(f"Using class ID {self.target_class_id}: '{self.model.names[self.target_class_id]}'")
        
        self.next_id = 1
        self.tracks = {}
        self.lost_tracks = {}
        self.player_features = {}
        self.player_colors = {}
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (255, 20, 147), (0, 191, 255),
            (50, 205, 50), (255, 69, 0), (138, 43, 226), (255, 105, 180), (32, 178, 170)
        ]
        
        self.frame_count = 0
        self.learning_phase = True
    
    def _find_target_class(self):
        target_names = ['person', 'player', 'people', 'human', 'man', 'woman', 'athlete']
        
        for i, name in self.model.names.items():
            if any(target in name.lower() for target in target_names):
                return i
        
        print("\nNo person/player class found automatically.")
        print("Available classes:")
        for i, name in self.model.names.items():
            print(f"  {i}: {name}")
        
        while True:
            try:
                choice = input("Enter class ID to track (or press Enter for class 0): ").strip()
                if choice == "":
                    return 0
                class_id = int(choice)
                if class_id in self.model.names:
                    return class_id
                else:
                    print(f"Invalid class ID. Choose from: {list(self.model.names.keys())}")
            except ValueError:
                print("Please enter a valid number.")
    
    def _extract_features(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.feature_dim)
        
        region = frame[y1:y2, x1:x2]
        region = cv2.resize(region, (64, 128))
        
        features = []
        
        for i in range(3):
            hist = cv2.calcHist([region], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_hist = np.histogram(grad_mag, bins=32, range=(0, 255))[0]
        features.extend(grad_hist)
        
        features = np.array(features, dtype=np.float32)
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _detect_players(self, frame):
        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i, (box, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                if int(cls.item()) == self.target_class_id:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    if x2 > x1 and y2 > y1:
                        features = self._extract_features(frame, [x1, y1, x2, y2])
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1+x2)//2, (y1+y2)//2],
                            'conf': conf.item(),
                            'features': features
                        })
        
        return detections
    
    def _cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def _calculate_similarity(self, det1, det2):
        app_sim = self._cosine_similarity(det1['features'], det2['features'])
        
        distance = np.sqrt((det1['center'][0] - det2['center'][0])**2 + 
                          (det1['center'][1] - det2['center'][1])**2)
        motion_sim = 1.0 / (1.0 + distance / 100.0)
        
        total_sim = (self.appearance_weight * app_sim + 
                    self.motion_weight * motion_sim)
        
        return total_sim, app_sim, motion_sim, distance
    
    def _update_player_features(self, track_id, features):
        if track_id not in self.player_features:
            self.player_features[track_id] = []
        
        self.player_features[track_id].append(features)
        
        max_features = 10
        if len(self.player_features[track_id]) > max_features:
            self.player_features[track_id] = self.player_features[track_id][-max_features:]
    
    def _get_player_average_features(self, track_id):
        if track_id not in self.player_features or not self.player_features[track_id]:
            return None
        
        features = np.array(self.player_features[track_id])
        return np.mean(features, axis=0)
    
    def _track_with_reidentification(self, detections):
        self.frame_count += 1
        
        if self.frame_count > self.learning_frames:
            self.learning_phase = False
        
        for track in self.tracks.values():
            track['age'] += 1
        
        active_tracks = {}
        for track_id, track in self.tracks.items():
            if track['age'] < self.max_age:
                active_tracks[track_id] = track
            else:
                self.lost_tracks[track_id] = {
                    'last_seen': self.frame_count,
                    'features': self._get_player_average_features(track_id),
                    'last_bbox': track['bbox'],
                    'last_center': track['center']
                }
        
        self.tracks = active_tracks
        
        self.lost_tracks = {
            k: v for k, v in self.lost_tracks.items() 
            if self.frame_count - v['last_seen'] < self.reid_memory_frames
        }
        
        if not detections:
            return list(self.tracks.values())
        
        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()
        
        if self.tracks:
            track_ids = list(self.tracks.keys())
            similarity_matrix = np.zeros((len(detections), len(track_ids)))
            
            for i, det in enumerate(detections):
                for j, track_id in enumerate(track_ids):
                    track = self.tracks[track_id]
                    similarity, _, _, distance = self._calculate_similarity(det, track)
                    
                    if distance < self.max_distance:
                        similarity_matrix[i, j] = similarity
            
            while True:
                best_sim = 0
                best_det = -1
                best_track = -1
                
                for i in range(len(detections)):
                    if i not in unmatched_detections:
                        continue
                    for j in range(len(track_ids)):
                        if track_ids[j] in matched_tracks:
                            continue
                        if similarity_matrix[i, j] > best_sim:
                            best_sim = similarity_matrix[i, j]
                            best_det = i
                            best_track = j
                
                if best_sim > 0.3:
                    track_id = track_ids[best_track]
                    det = detections[best_det]
                    
                    self.tracks[track_id].update({
                        'bbox': det['bbox'],
                        'center': det['center'],
                        'conf': det['conf'],
                        'features': det['features'],
                        'age': 0
                    })
                    
                    self._update_player_features(track_id, det['features'])
                    
                    unmatched_detections.remove(best_det)
                    matched_tracks.add(track_id)
                else:
                    break
        
        for i in list(unmatched_detections):
            det = detections[i]
            best_match = None
            best_similarity = 0
            
            for lost_id, lost_track in self.lost_tracks.items():
                if lost_track['features'] is not None:
                    app_sim = self._cosine_similarity(det['features'], lost_track['features'])
                    
                    if app_sim > best_similarity and app_sim > self.reid_threshold:
                        best_similarity = app_sim
                        best_match = lost_id
            
            if best_match is not None:
                print(f"Re-identified player {best_match} with similarity {best_similarity:.3f}")
                
                self.tracks[best_match] = {
                    'id': best_match,
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'conf': det['conf'],
                    'features': det['features'],
                    'age': 0
                }
                
                self._update_player_features(best_match, det['features'])
                
                del self.lost_tracks[best_match]
                unmatched_detections.remove(i)
        
        for i in unmatched_detections:
            det = detections[i]
            
            color_idx = (self.next_id - 1) % len(self.colors)
            self.player_colors[self.next_id] = self.colors[color_idx]
            
            self.tracks[self.next_id] = {
                'id': self.next_id,
                'bbox': det['bbox'],
                'center': det['center'],
                'conf': det['conf'],
                'features': det['features'],
                'age': 0
            }
            
            self._update_player_features(self.next_id, det['features'])
            
            self.next_id += 1
        
        return list(self.tracks.values())
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        print(f"Processing with re-identification: {width}x{height}, {total_frames} frames")
        print(f"Learning phase: first {self.learning_frames} frames")
        
        frame_num = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            detections = self._detect_players(frame)
            tracks = self._track_with_reidentification(detections)
            
            for track in tracks:
                if track['age'] < 10:
                    x1, y1, x2, y2 = track['bbox']
                    track_id = track['id']
                    
                    color = self.player_colors.get(track_id, (0, 255, 0))
                    
                    if self.learning_phase:
                        thickness = 3
                        text_color = (0, 255, 255)
                    else:
                        thickness = 2
                        text_color = color
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    label = f'P{track_id}'
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    conf_text = f'{track["conf"]:.2f}'
                    cv2.putText(frame, conf_text, (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
            status_y = 30
            cv2.putText(frame, f'Frame: {frame_num}/{total_frames}', (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            status_y += 25
            cv2.putText(frame, f'Active: {len(tracks)} | Lost: {len(self.lost_tracks)}', 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.learning_phase:
                status_y += 25
                cv2.putText(frame, 'LEARNING PHASE', (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            out.write(frame)
            
            if frame_num % max(1, int(fps)) == 0:
                elapsed = time.time() - start_time
                progress = (frame_num / total_frames) * 100
                fps_current = frame_num / elapsed if elapsed > 0 else 0
                eta = (elapsed / frame_num) * (total_frames - frame_num) if frame_num > 0 else 0
                
                print(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f} | "
                      f"Active: {len(tracks)} | Lost: {len(self.lost_tracks)} | "
                      f"ETA: {eta:.0f}s")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.1f}s @ {total_frames/total_time:.1f} FPS")
        print(f"Output: {self.output_path}")
        print(f"Total unique players tracked: {self.next_id - 1}")
        print(f"Final player features learned: {len(self.player_features)}")

if __name__ == "__main__":
    MODEL_PATH = 'best.pt'
    VIDEO_PATH = '15sec_input_720p.mp4'
    OUTPUT_PATH = 'output_reidentification.mp4'
    
    tracker = PlayerReidentificationTracker(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH)
    tracker.process_video()