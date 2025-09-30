import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

class MultiTimeScaleQueueMonitor:
    """
    Multi-time scale queue monitoring system to handle microburst-induced 
    non-linear and transient queue changes.
    
    Features:
    - Short-term monitoring for instantaneous steep increases
    - Long-term monitoring for gradual trends
    - Dual-threshold slope criteria
    - Adaptive threshold calibration
    - ECN-like early detection for burst suppression
    """
    
    def __init__(self, 
                 short_window_ms: float = 10.0,  # 10ms for short-term
                 long_window_ms: float = 1000.0,  # 1s for long-term
                 short_threshold_ratio: float = 0.3,  # 30% increase in short window
                 long_threshold_ratio: float = 0.1,  # 10% increase in long window
                 adaptation_window: int = 100,  # Last 100 samples for adaptation
                 min_samples_for_adaptation: int = 20):
        
        self.short_window_ms = short_window_ms
        self.long_window_ms = long_window_ms
        self.short_threshold_ratio = short_threshold_ratio
        self.long_threshold_ratio = long_threshold_ratio
        self.adaptation_window = adaptation_window
        self.min_samples_for_adaptation = min_samples_for_adaptation
        
        # Queue history storage: (timestamp_ms, queue_size)
        self.queue_history: deque = deque()
        self.slope_history: deque = deque()
        
        # Adaptive thresholds
        self.adaptive_short_threshold = short_threshold_ratio
        self.adaptive_long_threshold = long_threshold_ratio
        
        # Statistics for adaptation
        self.recent_slopes: deque = deque(maxlen=adaptation_window)
        self.recent_short_slopes: deque = deque(maxlen=adaptation_window)
        
        # Detection state
        self.last_detection_time = 0
        self.consecutive_detections = 0
        self.burst_active = False
        
    def add_queue_sample(self, queue_size: float, timestamp_ms: Optional[float] = None):
        """Add a new queue size sample"""
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000  # Convert to milliseconds
            
        self.queue_history.append((timestamp_ms, queue_size))
        
        # Keep only recent history (last 10 seconds)
        cutoff_time = timestamp_ms - 10000
        while self.queue_history and self.queue_history[0][0] < cutoff_time:
            self.queue_history.popleft()
            
        # Calculate and store slope if we have enough data
        if len(self.queue_history) >= 2:
            slope = self._calculate_slope()
            if slope is not None:
                self.slope_history.append((timestamp_ms, slope))
                
                # Keep slope history manageable
                while len(self.slope_history) > self.adaptation_window:
                    self.slope_history.popleft()
                    
                # Update adaptive thresholds
                self._update_adaptive_thresholds()
    
    def _calculate_slope(self) -> Optional[float]:
        """Calculate current queue growth rate"""
        if len(self.queue_history) < 2:
            return None
            
        # Use linear regression for slope calculation
        timestamps = np.array([item[0] for item in self.queue_history])
        queue_sizes = np.array([item[1] for item in self.queue_history])
        
        # Normalize timestamps to start from 0
        timestamps = timestamps - timestamps[0]
        
        if len(timestamps) < 2:
            return None
            
        # Linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        n = len(timestamps)
        sum_x = np.sum(timestamps)
        sum_y = np.sum(queue_sizes)
        sum_xy = np.sum(timestamps * queue_sizes)
        sum_x2 = np.sum(timestamps ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent history"""
        if len(self.slope_history) < self.min_samples_for_adaptation:
            return
            
        recent_slopes = [item[1] for item in self.slope_history]
        slope_mean = np.mean(recent_slopes)
        slope_std = np.std(recent_slopes)
        
        # Adaptive short-term threshold: slightly above normal jitter
        # but below half of real congestion growth rate
        noise_threshold = slope_mean + 2 * slope_std
        congestion_threshold = abs(slope_mean) + 3 * slope_std
        
        self.adaptive_short_threshold = max(
            self.short_threshold_ratio * 0.5,  # Minimum threshold
            min(noise_threshold, congestion_threshold * 0.5)
        )
        
        # Adaptive long-term threshold: more conservative
        self.adaptive_long_threshold = max(
            self.long_threshold_ratio * 0.5,
            noise_threshold * 0.7
        )
    
    def detect_congestion(self) -> Dict[str, any]:
        """
        Detect congestion using dual-threshold slope criteria.
        
        Returns:
            Dict with detection results including:
            - is_congested: bool
            - short_term_detected: bool  
            - long_term_detected: bool
            - confidence: float (0-1)
            - recommended_action: str
        """
        if len(self.queue_history) < 3:
            return {
                'is_congested': False,
                'short_term_detected': False,
                'long_term_detected': False,
                'confidence': 0.0,
                'recommended_action': 'continue_monitoring'
            }
        
        current_time = self.queue_history[-1][0]
        
        # Short-term detection: look for steep increases in recent window
        short_term_detected = self._detect_short_term_congestion(current_time)
        
        # Long-term detection: look for gradual trends
        long_term_detected = self._detect_long_term_congestion(current_time)
        
        # Determine overall congestion status
        is_congested = short_term_detected or long_term_detected
        
        # Calculate confidence based on detection consistency
        confidence = self._calculate_confidence(short_term_detected, long_term_detected)
        
        # Recommend action based on detection type
        recommended_action = self._recommend_action(short_term_detected, long_term_detected, confidence)
        
        # Update detection state
        if is_congested:
            self.consecutive_detections += 1
            self.burst_active = True
        else:
            self.consecutive_detections = 0
            self.burst_active = False
            
        return {
            'is_congested': is_congested,
            'short_term_detected': short_term_detected,
            'long_term_detected': long_term_detected,
            'confidence': confidence,
            'recommended_action': recommended_action,
            'consecutive_detections': self.consecutive_detections,
            'adaptive_short_threshold': self.adaptive_short_threshold,
            'adaptive_long_threshold': self.adaptive_long_threshold
        }
    
    def _detect_short_term_congestion(self, current_time: float) -> bool:
        """Detect short-term congestion (microbursts)"""
        window_start = current_time - self.short_window_ms
        
        # Get samples within short window
        recent_samples = [(t, q) for t, q in self.queue_history if t >= window_start]
        
        if len(recent_samples) < 2:
            return False
            
        # Calculate growth rate in short window
        timestamps = np.array([t for t, q in recent_samples])
        queue_sizes = np.array([q for t, q in recent_samples])
        
        if len(timestamps) < 2:
            return False
            
        # Calculate instantaneous growth rate
        time_diff = (timestamps[-1] - timestamps[0]) / 1000.0  # Convert to seconds
        queue_diff = queue_sizes[-1] - queue_sizes[0]
        
        if time_diff <= 0:
            return False
            
        growth_rate = queue_diff / time_diff
        
        # Check against adaptive threshold
        return growth_rate > self.adaptive_short_threshold
    
    def _detect_long_term_congestion(self, current_time: float) -> bool:
        """Detect long-term congestion trends"""
        window_start = current_time - self.long_window_ms
        
        # Get samples within long window
        recent_samples = [(t, q) for t, q in self.queue_history if t >= window_start]
        
        if len(recent_samples) < 3:
            return False
            
        # Use linear regression for trend analysis
        timestamps = np.array([t for t, q in recent_samples])
        queue_sizes = np.array([q for t, q in recent_samples])
        
        # Normalize timestamps
        timestamps = timestamps - timestamps[0]
        
        if len(timestamps) < 3:
            return False
            
        # Linear regression slope
        n = len(timestamps)
        sum_x = np.sum(timestamps)
        sum_y = np.sum(queue_sizes)
        sum_xy = np.sum(timestamps * queue_sizes)
        sum_x2 = np.sum(timestamps ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return False
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Check against adaptive long-term threshold
        return slope > self.adaptive_long_threshold
    
    def _calculate_confidence(self, short_term: bool, long_term: bool) -> float:
        """Calculate detection confidence based on multiple factors"""
        confidence = 0.0
        
        # Base confidence from detection types
        if short_term and long_term:
            confidence = 0.9  # High confidence when both detect
        elif short_term:
            confidence = 0.7  # Medium-high for short-term detection
        elif long_term:
            confidence = 0.6  # Medium for long-term detection
        
        # Boost confidence for consecutive detections
        if self.consecutive_detections > 0:
            confidence += min(0.2, self.consecutive_detections * 0.05)
            
        # Reduce confidence if thresholds are too sensitive
        if self.adaptive_short_threshold < self.short_threshold_ratio * 0.3:
            confidence *= 0.8
            
        return min(1.0, confidence)
    
    def _recommend_action(self, short_term: bool, long_term: bool, confidence: float) -> str:
        """Recommend action based on detection results"""
        if short_term and confidence > 0.8:
            return "immediate_ecn_marking"  # Immediate ECN marking for microbursts
        elif long_term and confidence > 0.7:
            return "gradual_rate_limiting"  # Gradual rate limiting for trends
        elif short_term or long_term:
            return "monitor_closely"  # Monitor closely but don't act yet
        else:
            return "continue_monitoring"  # Continue normal monitoring
    
    def get_queue_statistics(self) -> Dict[str, float]:
        """Get current queue statistics"""
        if not self.queue_history:
            return {
                'current_size': 0.0,
                'avg_size': 0.0,
                'max_size': 0.0,
                'growth_rate': 0.0,
                'variance': 0.0
            }
            
        queue_sizes = [q for _, q in self.queue_history]
        
        return {
            'current_size': queue_sizes[-1],
            'avg_size': np.mean(queue_sizes),
            'max_size': np.max(queue_sizes),
            'growth_rate': self._calculate_slope() or 0.0,
            'variance': np.var(queue_sizes),
            'samples_count': len(queue_sizes)
        }
    
    def reset(self):
        """Reset monitor state"""
        self.queue_history.clear()
        self.slope_history.clear()
        self.recent_slopes.clear()
        self.recent_short_slopes.clear()
        self.consecutive_detections = 0
        self.burst_active = False
        self.adaptive_short_threshold = self.short_threshold_ratio
        self.adaptive_long_threshold = self.long_threshold_ratio
