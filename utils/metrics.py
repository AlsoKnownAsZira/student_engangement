"""
Metrics calculation for engagement scoring
"""

import numpy as np
from collections import defaultdict, deque
from scipy.spatial import distance


class EngagementScorer:
    """Rule-based engagement scoring using pose keypoints"""
    
    # COCO keypoint indices
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    def __init__(self, weights=None, thresholds=None):
        """
        Initialize engagement scorer
        
        Args:
            weights: Dictionary of feature weights
            thresholds: Dictionary of engagement thresholds
        """
        # Default weights
        self.weights = weights or {
            'pose_upright': 0.3,
            'head_forward': 0.25,
            'hands_visible': 0.2,
            'body_stable': 0.15,
            'sitting': 0.1
        }
        
        # Default thresholds
        self.thresholds = thresholds or {
            'high': 0.65,
            'medium': 0.35
        }
        
        # History for temporal features
        self.position_history = defaultdict(lambda: deque(maxlen=30))
    
    def calculate_score(self, keypoints, track_id=None):
        """
        Calculate engagement score from pose keypoints
        
        Args:
            keypoints: Pose keypoints array (17, 3) - [x, y, confidence]
            track_id: Track ID for temporal analysis
        
        Returns:
            Engagement score (0-1)
        """
        if keypoints is None or len(keypoints) == 0:
            return 0.5  # default neutral score
        
        scores = {}
        
        # 1. Pose upright (shoulder-hip alignment)
        scores['pose_upright'] = self._score_pose_upright(keypoints)
        
        # 2. Head forward (facing camera)
        scores['head_forward'] = self._score_head_forward(keypoints)
        
        # 3. Hands visible (taking notes, gesturing)
        scores['hands_visible'] = self._score_hands_visible(keypoints)
        
        # 4. Body stable (not fidgeting) - needs temporal data
        if track_id is not None:
            scores['body_stable'] = self._score_body_stable(keypoints, track_id)
        else:
            scores['body_stable'] = 0.5
        
        # 5. Sitting (appropriate posture)
        scores['sitting'] = self._score_sitting(keypoints)
        
        # Weighted sum
        total_score = sum(
            scores[feat] * self.weights[feat] 
            for feat in scores
        )
        
        return total_score
    
    def _score_pose_upright(self, keypoints):
        """Score based on upright posture"""
        try:
            # Get shoulder and hip points
            l_shoulder = keypoints[self.KEYPOINT_INDICES['left_shoulder']]
            r_shoulder = keypoints[self.KEYPOINT_INDICES['right_shoulder']]
            l_hip = keypoints[self.KEYPOINT_INDICES['left_hip']]
            r_hip = keypoints[self.KEYPOINT_INDICES['right_hip']]
            
            # Check confidence
            if any(kp[2] < 0.3 for kp in [l_shoulder, r_shoulder, l_hip, r_hip]):
                return 0.5
            
            # Calculate shoulder and hip centers
            shoulder_center = (l_shoulder[:2] + r_shoulder[:2]) / 2
            hip_center = (l_hip[:2] + r_hip[:2]) / 2
            
            # Calculate angle from vertical
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            angle = np.abs(np.arctan2(dx, dy))  # angle from vertical
            
            # Score: closer to vertical (0) = higher score
            score = max(0, 1 - angle / (np.pi / 4))  # normalize to 45 degrees
            return score
            
        except Exception:
            return 0.5
    
    def _score_head_forward(self, keypoints):
        """Score based on head facing forward"""
        try:
            nose = keypoints[self.KEYPOINT_INDICES['nose']]
            l_eye = keypoints[self.KEYPOINT_INDICES['left_eye']]
            r_eye = keypoints[self.KEYPOINT_INDICES['right_eye']]
            l_ear = keypoints[self.KEYPOINT_INDICES['left_ear']]
            r_ear = keypoints[self.KEYPOINT_INDICES['right_ear']]
            
            if any(kp[2] < 0.3 for kp in [nose, l_eye, r_eye]):
                return 0.5
            
            # Both eyes visible = facing forward
            eyes_visible = (l_eye[2] > 0.3 and r_eye[2] > 0.3)
            
            # Ears less visible than eyes = facing forward
            ears_visible = (l_ear[2] > 0.3 or r_ear[2] > 0.3)
            
            if eyes_visible and not ears_visible:
                return 1.0
            elif eyes_visible:
                return 0.7
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _score_hands_visible(self, keypoints):
        """Score based on hand visibility (taking notes, gesturing)"""
        try:
            l_wrist = keypoints[self.KEYPOINT_INDICES['left_wrist']]
            r_wrist = keypoints[self.KEYPOINT_INDICES['right_wrist']]
            l_elbow = keypoints[self.KEYPOINT_INDICES['left_elbow']]
            r_elbow = keypoints[self.KEYPOINT_INDICES['right_elbow']]
            
            # Count visible hands
            hands_visible = sum([
                l_wrist[2] > 0.3,
                r_wrist[2] > 0.3
            ])
            
            # Hands in front of body (y-coordinate)
            l_shoulder = keypoints[self.KEYPOINT_INDICES['left_shoulder']]
            r_shoulder = keypoints[self.KEYPOINT_INDICES['right_shoulder']]
            
            if l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3:
                shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                
                # Hands above desk level (below shoulders)
                hands_active = sum([
                    l_wrist[2] > 0.3 and l_wrist[1] > shoulder_y,
                    r_wrist[2] > 0.3 and r_wrist[1] > shoulder_y
                ])
                
                return (hands_visible * 0.5 + hands_active * 0.5) / 2
            
            return hands_visible / 2
            
        except Exception:
            return 0.5
    
    def _score_body_stable(self, keypoints, track_id):
        """Score based on body stability (not fidgeting)"""
        try:
            # Use nose as reference point
            nose = keypoints[self.KEYPOINT_INDICES['nose']]
            
            if nose[2] < 0.3:
                return 0.5
            
            # Add to history
            self.position_history[track_id].append(nose[:2])
            
            # Need at least 10 frames for stability check
            if len(self.position_history[track_id]) < 10:
                return 0.5
            
            # Calculate movement variance
            positions = np.array(list(self.position_history[track_id]))
            variance = np.var(positions, axis=0).sum()
            
            # Lower variance = more stable = higher score
            # Normalize (typical variance range: 0-1000)
            score = max(0, 1 - variance / 1000)
            return score
            
        except Exception:
            return 0.5
    
    def _score_sitting(self, keypoints):
        """Score based on sitting posture"""
        try:
            # Check hip and knee visibility
            l_hip = keypoints[self.KEYPOINT_INDICES['left_hip']]
            r_hip = keypoints[self.KEYPOINT_INDICES['right_hip']]
            l_knee = keypoints[self.KEYPOINT_INDICES['left_knee']]
            r_knee = keypoints[self.KEYPOINT_INDICES['right_knee']]
            
            if all(kp[2] < 0.3 for kp in [l_hip, r_hip]):
                return 0.5
            
            # Hips visible, knees visible = sitting
            hips_visible = (l_hip[2] > 0.3 or r_hip[2] > 0.3)
            knees_visible = (l_knee[2] > 0.3 or r_knee[2] > 0.3)
            
            if hips_visible and knees_visible:
                return 1.0
            elif hips_visible:
                return 0.7
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def get_engagement_level(self, score):
        """Convert score to engagement level"""
        if score >= self.thresholds['high']:
            return 'high'
        elif score >= self.thresholds['medium']:
            return 'medium'
        else:
            return 'low'


class EngagementMetrics:
    """Calculate aggregate engagement metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.scores = []
        self.levels = []
        self.per_student = defaultdict(list)
    
    def add_frame(self, frame_scores):
        """
        Add scores from one frame
        
        Args:
            frame_scores: Dict {track_id: (score, level)}
        """
        for track_id, (score, level) in frame_scores.items():
            self.scores.append(score)
            self.levels.append(level)
            self.per_student[track_id].append((score, level))
    
    def get_class_summary(self):
        """Get summary statistics for entire class"""
        if not self.scores:
            return {}
        
        scores = np.array(self.scores)
        
        # Level counts
        level_counts = {
            'high': self.levels.count('high'),
            'medium': self.levels.count('medium'),
            'low': self.levels.count('low')
        }
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'median_score': np.median(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'level_counts': level_counts,
            'level_percentages': {
                k: v / len(self.levels) * 100 
                for k, v in level_counts.items()
            },
            'total_observations': len(self.scores),
            'num_students': len(self.per_student)
        }
    
    def get_student_summary(self, track_id):
        """Get summary for specific student"""
        if track_id not in self.per_student:
            return {}
        
        data = self.per_student[track_id]
        scores = [s for s, l in data]
        levels = [l for s, l in data]
        
        level_counts = {
            'high': levels.count('high'),
            'medium': levels.count('medium'),
            'low': levels.count('low')
        }
        
        return {
            'track_id': track_id,
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'dominant_level': max(level_counts, key=level_counts.get),
            'level_counts': level_counts,
            'total_observations': len(scores)
        }
    
    def get_all_students_summary(self):
        """Get summary for all students"""
        return {
            track_id: self.get_student_summary(track_id)
            for track_id in self.per_student.keys()
        }