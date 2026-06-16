"""
Video Processor — extracts frames from video files and runs full analysis.
Provides per-frame results and aggregated video-level summary.
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Callable
import time
from collections import Counter


class VideoProcessor:
    """
    Processes video files by extracting frames at regular intervals
    and running the full VisionAnalyzer pipeline on each frame.
    Produces both per-frame results and an aggregated video summary.
    """

    def __init__(self, analyzer, max_frames: int = 20, fps_target: float = 1.0):
        """
        analyzer: VisionAnalyzer instance
        max_frames: maximum number of frames to analyze (for performance)
        fps_target: target frames per second to sample (e.g. 1.0 = every 1 second)
        """
        self.analyzer = analyzer
        self.max_frames = max_frames
        self.fps_target = fps_target

    def process_video(
        self,
        video_path: str,
        progress_cb: Optional[Callable] = None
    ) -> Dict:
        """
        Main entry point for video analysis.
        Returns comprehensive video analysis result.
        """
        start_time = time.time()

        def _progress(step: str, pct: int):
            if progress_cb:
                progress_cb(step, pct)

        _progress("Opening video file...", 2)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0

        metadata = {
            "duration_seconds": round(duration_sec, 2),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "aspect_ratio": self._compute_aspect_ratio(width, height)
        }

        _progress("Extracting frames...", 5)

        # Determine frame sampling strategy
        sample_interval = max(1, int(fps / self.fps_target))
        frames_to_sample = list(range(0, total_frames, sample_interval))

        # Limit to max_frames
        if len(frames_to_sample) > self.max_frames:
            step = len(frames_to_sample) // self.max_frames
            frames_to_sample = frames_to_sample[::step][:self.max_frames]

        # Extract frames
        frames = []
        frame_timestamps = []

        for frame_idx in frames_to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                frame_timestamps.append(round(frame_idx / fps, 2))

        cap.release()

        total = len(frames)
        if total == 0:
            raise ValueError("No frames could be extracted from the video.")

        # Analyze each frame
        frame_results = []
        all_labels = []
        all_scenes = []
        all_moods = []

        for i, (frame, ts) in enumerate(zip(frames, frame_timestamps)):
            pct = int(10 + (i / total) * 80)
            _progress(f"Analyzing frame {i+1}/{total} (t={ts}s)...", pct)

            try:
                frame_result = self.analyzer.analyze_image(frame)
                frame_result["timestamp"] = ts
                frame_result["frame_index"] = i

                # Collect for aggregate analysis
                for det in frame_result.get("detections", []):
                    all_labels.append(det["label"])
                if frame_result.get("scene_classifications"):
                    all_scenes.append(frame_result["scene_classifications"][0]["label"])
                if frame_result.get("mood"):
                    all_moods.append(frame_result["mood"])

                frame_results.append(frame_result)
            except Exception as e:
                print(f"[VideoProcessor] Error on frame {i}: {e}")
                frame_results.append({
                    "timestamp": ts,
                    "frame_index": i,
                    "error": str(e),
                    "detections": []
                })

        _progress("Generating video summary...", 92)

        # Aggregate statistics
        label_counter = Counter(all_labels)
        scene_counter = Counter(all_scenes)
        mood_counter = Counter(all_moods)

        top_objects = [
            {"label": label, "count": count, "frequency_percent": round(count / max(len(frame_results), 1) * 100, 1)}
            for label, count in label_counter.most_common(10)
        ]

        # Build timeline events (changes in scene/key detections)
        timeline = self._build_timeline(frame_results)

        # Motion analysis (basic — based on frame differences)
        motion_scores = self._compute_motion(frames)

        video_summary = {
            "frames_analyzed": total,
            "dominant_scene": scene_counter.most_common(1)[0][0] if scene_counter else "Unknown",
            "dominant_mood": mood_counter.most_common(1)[0][0] if mood_counter else "Unknown",
            "top_objects": top_objects,
            "unique_objects_seen": len(label_counter),
            "total_detections": len(all_labels),
            "scene_variety": len(scene_counter),
            "motion_level": self._classify_motion(motion_scores),
            "motion_score": round(np.mean(motion_scores) if motion_scores else 0, 1),
            "analysis_time_seconds": round(time.time() - start_time, 2)
        }

        _progress("Video analysis complete!", 100)

        return {
            "type": "video",
            "metadata": metadata,
            "summary": video_summary,
            "timeline": timeline,
            "frame_results": frame_results
        }

    def _compute_motion(self, frames: List[Image.Image]) -> List[float]:
        """Compute per-frame motion score by comparing consecutive frames."""
        if len(frames) < 2:
            return [0.0]

        scores = []
        size = (160, 90)
        prev_gray = None

        for frame in frames:
            small = np.array(frame.convert("L").resize(size))
            if prev_gray is not None:
                diff = np.abs(small.astype(float) - prev_gray.astype(float))
                scores.append(float(diff.mean()))
            prev_gray = small

        return scores

    def _classify_motion(self, scores: List[float]) -> str:
        if not scores:
            return "unknown"
        mean_score = np.mean(scores)
        if mean_score < 2:
            return "static (minimal motion)"
        elif mean_score < 8:
            return "slow (gentle movement)"
        elif mean_score < 20:
            return "moderate motion"
        elif mean_score < 40:
            return "active (significant movement)"
        else:
            return "high motion (fast-paced)"

    def _build_timeline(self, frame_results: List[Dict]) -> List[Dict]:
        """Build a simplified timeline of key events/changes."""
        timeline = []
        prev_objects = set()

        for fr in frame_results:
            ts = fr.get("timestamp", 0)
            current_objects = set(
                d["label"] for d in fr.get("detections", [])
            )

            new_objects = current_objects - prev_objects
            lost_objects = prev_objects - current_objects

            if new_objects:
                timeline.append({
                    "timestamp": ts,
                    "event": "new_objects_appeared",
                    "objects": list(new_objects)
                })
            if lost_objects and prev_objects:
                timeline.append({
                    "timestamp": ts,
                    "event": "objects_left_frame",
                    "objects": list(lost_objects)
                })

            if fr.get("detections"):
                prev_objects = current_objects

        return timeline[:50]  # Limit timeline events

    def _compute_aspect_ratio(self, w: int, h: int) -> str:
        from math import gcd
        g = gcd(w, h)
        return f"{w//g}:{h//g}"
