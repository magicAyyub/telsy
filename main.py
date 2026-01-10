"""Real-time semantic segmentation using webcam or video file.
"""

import os
import yaml
import torch
import numpy as np
import cv2
import time
from typing import Tuple, Optional, Dict, Any, List
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
from bisenetv2_model import BiSeNetV2
from helper import get_device


class RealtimeSegmentation:
    def __init__(self, config_path: str = 'configs/config.yaml') -> None:
        with open(config_path, 'r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        
        self.device = get_device()
        self.model = self._load_model()
        self.preprocess = self._get_preprocessing()
        self.color_palette: np.ndarray = np.array(self.config['color_palette'], dtype=np.uint8)
        self.class_names: List[str] = self.config['class_names']
        
    def _load_model(self) -> BiSeNetV2:
        model_config: Dict[str, Any] = self.config['model']
        model: BiSeNetV2 = BiSeNetV2(n_classes=self.config['dataset']['num_classes'], aux_mode='eval')
        
        if os.path.exists(model_config['pretrained_path']):
            state_dict = torch.load(model_config['pretrained_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        return model
    
    def _get_preprocessing(self) -> transforms.Compose:
        image_size: Tuple[int, int] = tuple(self.config['dataset']['image_size'])
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame and return segmentation mask."""
        h: int
        w: int
        h, w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize to original size
        prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return prediction
    
    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to colored image."""
        h: int
        w: int
        h, w = mask.shape
        colored: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(len(self.color_palette)):
            colored[mask == class_id] = self.color_palette[class_id]
        
        return colored
    
    def overlay_mask(self, frame: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Overlay segmentation mask on original frame using torchvision."""
        h: int
        w: int
        h, w = frame.shape[:2]
        
        # Convert frame from BGR (OpenCV) to RGB tensor
        frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor: torch.Tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
        
        # Create boolean masks for each class
        num_classes: int = len(self.color_palette)
        masks: torch.Tensor = torch.zeros((num_classes, h, w), dtype=torch.bool)
        for class_id in range(num_classes):
            masks[class_id] = torch.from_numpy(mask == class_id)
        
        # Convert color palette to list of tuples
        colors: List[Tuple[int, int, int]] = [tuple(color) for color in self.color_palette]
        
        # Draw segmentation masks
        overlay_tensor: torch.Tensor = draw_segmentation_masks(
            frame_tensor,
            masks,
            alpha=1-alpha,
            colors=colors
        )
        
        # Convert back to BGR for OpenCV
        overlay_rgb: np.ndarray = overlay_tensor.permute(1, 2, 0).numpy()
        overlay_bgr: np.ndarray = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        
        return overlay_bgr
    
    def add_fps_text(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Add FPS counter to frame."""
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    
    def run_webcam(self, camera_id: int = 0, show_overlay: bool = True) -> None:
        """Run real-time segmentation on webcam feed."""
        cap: cv2.VideoCapture = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        print("Real-time segmentation started. Press 'q' to quit.")
        print("Press 's' to toggle segmentation view")
        print("Press 'o' to toggle overlay")
        
        show_segmentation: bool = True
        
        while True:
            start_time: float = time.time()
            
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            mask = self.predict_frame(frame)
            colored_mask = self.colorize_mask(mask)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            
            # Prepare display
            if show_segmentation:
                if show_overlay:
                    display = self.overlay_mask(frame, mask)
                else:
                    display = colored_mask
            else:
                display = frame
            
            display = self.add_fps_text(display, fps)
            
            cv2.imshow('Real-time Segmentation', display)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_segmentation = not show_segmentation
            elif key == ord('o'):
                show_overlay = not show_overlay
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_stream(self, stream_url: str, save_output: bool = False, output_path: Optional[str] = None) -> None:
        """Run real-time segmentation on RTSP/RTMP stream (e.g., from Larix broadcaster).
        
        Args:
            stream_url: RTSP or RTMP stream URL (e.g., 'rtsp://192.168.1.100:8554/live')
            save_output: Whether to save the output to a video file
            output_path: Path to save the output video (required if save_output=True)
        """
        cap: cv2.VideoCapture = cv2.VideoCapture(stream_url)
        
        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"Error: Cannot open stream {stream_url}")
            print("Make sure:")
            print("  1. Larix broadcaster is streaming")
            print("  2. The stream URL is correct")
            print("  3. Your computer can access the stream (same network)")
            return
        
        # Get stream properties
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps: int = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if not available
        
        # Setup video writer if saving
        out: Optional[cv2.VideoWriter] = None
        if save_output and output_path:
            fourcc: int = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Recording to: {output_path}")
        
        print(f"\nStream started: {stream_url}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Toggle segmentation view")
        print("  'o' - Toggle overlay mode")
        print("  'r' - Toggle recording (if output path specified)\n")
        
        show_segmentation: bool = True
        show_overlay: bool = True
        recording: bool = save_output
        frame_count: int = 0
        
        while True:
            start_time: float = time.time()
            
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            
            if not ret:
                print("\nStream ended or connection lost. Retrying...")
                time.sleep(1)
                continue
            
            frame_count += 1
            
            # Predict
            mask: np.ndarray = self.predict_frame(frame)
            colored_mask: np.ndarray = self.colorize_mask(mask)
            
            # Calculate FPS
            fps_current: float = 1.0 / max(time.time() - start_time, 0.001)
            
            # Prepare display
            display: np.ndarray
            if show_segmentation:
                if show_overlay:
                    display = self.overlay_mask(frame, mask)
                else:
                    display = colored_mask
            else:
                display = frame
            
            # Add FPS and recording indicator
            display = self.add_fps_text(display, fps_current)
            if recording and out is not None:
                cv2.putText(display, 'REC', (display.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save frame if recording
            if recording and out is not None:
                out.write(display)
            
            cv2.imshow('Live Stream Segmentation', display)
            
            # Key handling
            key: int = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_segmentation = not show_segmentation
                print(f"Segmentation: {'ON' if show_segmentation else 'OFF'}")
            elif key == ord('o'):
                show_overlay = not show_overlay
                print(f"Overlay mode: {'ON' if show_overlay else 'OFF'}")
            elif key == ord('r') and output_path:
                recording = not recording
                if recording and out is None:
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                print(f"Recording: {'ON' if recording else 'OFF'}")
        
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nStream processing complete. Frames processed: {frame_count}")
        if output_path and save_output:
            print(f"Video saved to: {output_path}")
    
    def run_video(self, video_path: str, output_path: Optional[str] = None, show_overlay: bool = True) -> None:
        """Run segmentation on video file."""
        cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps: int = int(cap.get(cv2.CAP_PROP_FPS))
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        out: Optional[cv2.VideoWriter] = None
        if output_path:
            fourcc: int = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_count: int = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Predict
            mask = self.predict_frame(frame)
            colored_mask = self.colorize_mask(mask)
            
            # Create output frame
            if show_overlay:
                output_frame = self.overlay_mask(frame, mask)
            else:
                output_frame = colored_mask
            
            # Calculate FPS
            inference_fps = 1.0 / (time.time() - start_time)
            output_frame = self.add_fps_text(output_frame, inference_fps)
            
            # Write frame
            if output_path:
                out.write(output_frame)
            
            # Display progress
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - FPS: {inference_fps:.1f}")
            
            # Show preview
            cv2.imshow('Video Processing', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing complete. Output saved to: {output_path}")


def main() -> None:
    import argparse
    
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Real-time semantic segmentation')
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'video', 'stream'],
                        help='Run mode: webcam, video, or stream (RTSP/RTMP)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input video path (for video mode) or stream URL (for stream mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (for video/stream mode)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID (for webcam mode)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Config file path')
    parser.add_argument('--no-overlay', action='store_true',
                        help='Show only segmentation mask')
    parser.add_argument('--save-stream', action='store_true',
                        help='Save stream output to video file (for stream mode)')
    
    args: argparse.Namespace = parser.parse_args()
    
    segmenter: RealtimeSegmentation = RealtimeSegmentation(args.config)
    
    if args.mode == 'webcam':
        segmenter.run_webcam(camera_id=args.camera, show_overlay=not args.no_overlay)
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        segmenter.run_video(args.input, args.output, show_overlay=not args.no_overlay)
    elif args.mode == 'stream':
        if not args.input:
            print("Error: --input required for stream mode")
            print("\nExample URLs:")
            print("  RTSP: rtsp://192.168.1.100:8554/live")
            print("  RTMP: rtmp://192.168.1.100:1935/live")
            print("\nLarix broadcaster setup:")
            print("  1. Open Larix on your iPhone")
            print("  2. Go to Settings > Connections")
            print("  3. Add new connection (RTSP or RTMP)")
            print("  4. Use your computer's IP address")
            print("  5. Start streaming from the app")
            return
        segmenter.run_stream(args.input, save_output=args.save_stream, output_path=args.output)


if __name__ == '__main__':
    main()
