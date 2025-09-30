import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

class EnhancedMicroburstDetector:
    """
    Enhanced microburst detection system that addresses the non-linear and 
    transient nature of queue changes caused by microbursts.
    
    Key features:
    1. Multi-time scale evaluation (short-term vs long-term)
    2. Dual-threshold slope criteria
    3. Adaptive threshold calibration
    4. ECN-like early detection for burst suppression
    5. Handles cases where queue rises then falls rapidly (non-linear behavior)
    """
    
    def __init__(self, 
                 short_window_ms: float = 5.0,      # 5ms for microburst detection
                 long_window_ms: float = 500.0,      # 500ms for trend analysis
                 microburst_threshold: float = 0.4,  # 40% increase in short window
                 trend_threshold: float = 0.15,      # 15% increase in long window
                 adaptation_samples: int = 50,       # Samples for threshold adaptation
                 min_samples_for_adaptation: int = 10):
        
        self.short_window_ms = short_window_ms
        self.long_window_ms = long_window_ms
        self.microburst_threshold = microburst_threshold
        self.trend_threshold = trend_threshold
        self.adaptation_samples = adaptation_samples
        self.min_samples_for_adaptation = min_samples_for_adaptation
        
        # Queue history: (timestamp_ms, queue_size, utilization)
        self.queue_history: deque = deque()
        self.slope_history: deque = deque()
        
        # Adaptive thresholds (will be calibrated based on network history)
        self.adaptive_microburst_threshold = microburst_threshold
        self.adaptive_trend_threshold = trend_threshold
        
        # Statistics for threshold adaptation
        self.recent_slopes: deque = deque(maxlen=adaptation_samples)
        self.recent_microburst_slopes: deque = deque(maxlen=adaptation_samples)
        
        # Detection state
        self.consecutive_detections = 0
        self.burst_active = False
        self.last_peak_time = 0
        self.peak_queue_size = 0
        
        # Network characteristics for calibration
        self.network_jitter_std = 0.0
        self.normal_growth_rate = 0.0
        
    def add_queue_sample(self, queue_size: float, utilization: float = 0.0, 
                        timestamp_ms: Optional[float] = None):
        """Add a new queue size sample with utilization context"""
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000
            
        self.queue_history.append((timestamp_ms, queue_size, utilization))
        
        # Keep only recent history (last 5 seconds)
        cutoff_time = timestamp_ms - 5000
        while self.queue_history and self.queue_history[0][0] < cutoff_time:
            self.queue_history.popleft()
            
        # Calculate slope if we have enough data
        if len(self.queue_history) >= 2:
            slope = self._calculate_instantaneous_slope()
            if slope is not None:
                self.slope_history.append((timestamp_ms, slope))
                
                # Keep slope history manageable
                while len(self.slope_history) > self.adaptation_samples:
                    self.slope_history.popleft()
                    
                # Update adaptive thresholds
                self._calibrate_adaptive_thresholds()
                
        # Track peak detection for microburst analysis
        self._update_peak_tracking(queue_size, timestamp_ms)
    
    def _calculate_instantaneous_slope(self) -> Optional[float]:
        """Calculate instantaneous queue growth rate using linear regression"""
        if len(self.queue_history) < 2:
            return None
            
        timestamps = np.array([item[0] for item in self.queue_history])
        queue_sizes = np.array([item[1] for item in self.queue_history])
        
        # Normalize timestamps to start from 0
        timestamps = timestamps - timestamps[0]
        
        if len(timestamps) < 2:
            return None
            
        # Linear regression for slope calculation
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
    
    def _update_peak_tracking(self, queue_size: float, timestamp_ms: float):
        """Track queue peaks for microburst detection"""
        if queue_size > self.peak_queue_size:
            self.peak_queue_size = queue_size
            self.last_peak_time = timestamp_ms
        else:
            # Check if we're in a declining phase after a peak
            time_since_peak = timestamp_ms - self.last_peak_time
            if time_since_peak > self.short_window_ms:
                # Reset peak tracking if enough time has passed
                self.peak_queue_size = queue_size
                self.last_peak_time = timestamp_ms
    
    def _calibrate_adaptive_thresholds(self):
        """Calibrate adaptive thresholds based on network history"""
        if len(self.slope_history) < self.min_samples_for_adaptation:
            return
            
        recent_slopes = [item[1] for item in self.slope_history]
        slope_mean = np.mean(recent_slopes)
        slope_std = np.std(recent_slopes)
        
        # Update network characteristics
        self.network_jitter_std = slope_std
        self.normal_growth_rate = slope_mean
        
        # Adaptive microburst threshold: above normal jitter but below real congestion
        # This addresses the issue where normal jitter might trigger false positives
        noise_threshold = slope_mean + 2.5 * slope_std  # Above normal jitter
        congestion_threshold = abs(slope_mean) + 4 * slope_std  # Real congestion level
        
        self.adaptive_microburst_threshold = max(
            self.microburst_threshold * 0.3,  # Minimum threshold
            min(noise_threshold, congestion_threshold * 0.6)  # Below half of real congestion
        )
        
        # Adaptive trend threshold: more conservative for long-term trends
        self.adaptive_trend_threshold = max(
            self.trend_threshold * 0.5,
            noise_threshold * 0.8
        )
    
    def detect_microburst_congestion(self) -> Dict[str, any]:
        """
        Detect microburst-induced congestion using dual-threshold criteria.
        
        This method specifically addresses the issue where queues may rise rapidly
        then fall, making simple average growth rate calculations misleading.
        """
        if len(self.queue_history) < 3:
            return {
                'is_congested': False,
                'microburst_detected': False,
                'trend_detected': False,
                'confidence': 0.0,
                'recommended_action': 'continue_monitoring',
                'peak_detected': False,
                'non_linear_behavior': False
            }
        
        current_time = self.queue_history[-1][0]
        current_queue = self.queue_history[-1][1]
        
        # Short-term microburst detection
        microburst_detected = self._detect_short_term_microburst(current_time)
        
        # Long-term trend detection
        trend_detected = self._detect_long_term_trend(current_time)
        
        # Non-linear behavior detection (rise then fall)
        non_linear_behavior = self._detect_non_linear_behavior(current_time)
        
        # Peak detection
        peak_detected = self._detect_recent_peak(current_time)
        
        # Overall congestion determination
        is_congested = microburst_detected or trend_detected or non_linear_behavior
        
        # Calculate confidence
        confidence = self._calculate_detection_confidence(
            microburst_detected, trend_detected, non_linear_behavior, peak_detected
        )
        
        # Recommend action
        recommended_action = self._recommend_congestion_action(
            microburst_detected, trend_detected, non_linear_behavior, confidence
        )
        
        # Update detection state
        if is_congested:
            self.consecutive_detections += 1
            self.burst_active = True
        else:
            self.consecutive_detections = 0
            self.burst_active = False
            
        return {
            'is_congested': is_congested,
            'microburst_detected': microburst_detected,
            'trend_detected': trend_detected,
            'non_linear_behavior': non_linear_behavior,
            'peak_detected': peak_detected,
            'confidence': confidence,
            'recommended_action': recommended_action,
            'consecutive_detections': self.consecutive_detections,
            'adaptive_microburst_threshold': self.adaptive_microburst_threshold,
            'adaptive_trend_threshold': self.adaptive_trend_threshold,
            'network_jitter_std': self.network_jitter_std,
            'normal_growth_rate': self.normal_growth_rate
        }
    
    def _detect_short_term_microburst(self, current_time: float) -> bool:
        """Detect short-term microbursts (steep increases in queue size)"""
        window_start = current_time - self.short_window_ms
        
        # Get samples within short window
        recent_samples = [(t, q, u) for t, q, u in self.queue_history if t >= window_start]
        
        if len(recent_samples) < 2:
            return False
            
        # Calculate growth rate in short window
        timestamps = np.array([t for t, q, u in recent_samples])
        queue_sizes = np.array([q for t, q, u in recent_samples])
        
        if len(timestamps) < 2:
            return False
            
        # Calculate instantaneous growth rate
        time_diff = (timestamps[-1] - timestamps[0]) / 1000.0  # Convert to seconds
        queue_diff = queue_sizes[-1] - queue_sizes[0]
        
        if time_diff <= 0:
            return False
            
        growth_rate = queue_diff / time_diff
        
        # Check against adaptive threshold
        return growth_rate > self.adaptive_microburst_threshold
    
    def _detect_long_term_trend(self, current_time: float) -> bool:
        """Detect long-term congestion trends"""
        window_start = current_time - self.long_window_ms
        
        # Get samples within long window
        recent_samples = [(t, q, u) for t, q, u in self.queue_history if t >= window_start]
        
        if len(recent_samples) < 3:
            return False
            
        # Use linear regression for trend analysis
        timestamps = np.array([t for t, q, u in recent_samples])
        queue_sizes = np.array([q for t, q, u in recent_samples])
        
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
        return slope > self.adaptive_trend_threshold
    
    def _detect_non_linear_behavior(self, current_time: float) -> bool:
        """
        Detect non-linear behavior where queue rises then falls rapidly.
        This addresses the specific issue mentioned in the requirements.
        """
        window_start = current_time - self.short_window_ms
        
        # Get samples within short window
        recent_samples = [(t, q, u) for t, q, u in self.queue_history if t >= window_start]
        
        if len(recent_samples) < 3:
            return False
            
        queue_sizes = [q for t, q, u in recent_samples]
        
        # Check for rise-then-fall pattern
        if len(queue_sizes) >= 3:
            # Find peak in the window
            peak_idx = np.argmax(queue_sizes)
            
            # Check if there's a significant rise before peak
            if peak_idx > 0:
                rise_before_peak = queue_sizes[peak_idx] - queue_sizes[0]
                # Check if there's a significant fall after peak
                if peak_idx < len(queue_sizes) - 1:
                    fall_after_peak = queue_sizes[peak_idx] - queue_sizes[-1]
                    
                    # Non-linear behavior: significant rise AND fall
                    return (rise_before_peak > self.adaptive_microburst_threshold * 0.5 and
                            fall_after_peak > self.adaptive_microburst_threshold * 0.3)
        
        return False
    
    def _detect_recent_peak(self, current_time: float) -> bool:
        """Detect if there was a recent peak in queue size"""
        time_since_peak = current_time - self.last_peak_time
        return time_since_peak <= self.short_window_ms * 2  # Within 2x short window
    
    def _calculate_detection_confidence(self, microburst: bool, trend: bool, 
                                       non_linear: bool, peak: bool) -> float:
        """Calculate detection confidence based on multiple factors"""
        confidence = 0.0
        
        # Base confidence from detection types
        if microburst and trend:
            confidence = 0.95  # Very high confidence
        elif microburst and non_linear:
            confidence = 0.9   # High confidence for microburst + non-linear
        elif microburst:
            confidence = 0.8   # High confidence for microburst
        elif trend:
            confidence = 0.7   # Medium-high for trend
        elif non_linear:
            confidence = 0.6   # Medium for non-linear behavior
        
        # Boost confidence for consecutive detections
        if self.consecutive_detections > 0:
            confidence += min(0.15, self.consecutive_detections * 0.03)
            
        # Boost confidence if peak was recent
        if peak:
            confidence += 0.1
            
        # Reduce confidence if thresholds are too sensitive
        if self.adaptive_microburst_threshold < self.microburst_threshold * 0.2:
            confidence *= 0.9
            
        return min(1.0, confidence)
    
    def _recommend_congestion_action(self, microburst: bool, trend: bool, 
                                   non_linear: bool, confidence: float) -> str:
        """Recommend action based on detection results"""
        if microburst and confidence > 0.85:
            return "immediate_ecn_marking"  # Immediate ECN for microbursts
        elif non_linear and confidence > 0.8:
            return "adaptive_rate_limiting"  # Adaptive rate limiting for non-linear
        elif trend and confidence > 0.75:
            return "gradual_rate_limiting"  # Gradual rate limiting for trends
        elif microburst or non_linear:
            return "monitor_closely"  # Monitor closely
        else:
            return "continue_monitoring"  # Continue normal monitoring
    
    def get_detailed_statistics(self) -> Dict[str, float]:
        """Get detailed queue and detection statistics"""
        if not self.queue_history:
            return {
                'current_queue_size': 0.0,
                'avg_queue_size': 0.0,
                'max_queue_size': 0.0,
                'current_growth_rate': 0.0,
                'queue_variance': 0.0,
                'peak_queue_size': 0.0,
                'time_since_peak_ms': 0.0
            }
            
        queue_sizes = [q for _, q, _ in self.queue_history]
        current_time = self.queue_history[-1][0]
        
        return {
            'current_queue_size': queue_sizes[-1],
            'avg_queue_size': np.mean(queue_sizes),
            'max_queue_size': np.max(queue_sizes),
            'current_growth_rate': self._calculate_instantaneous_slope() or 0.0,
            'queue_variance': np.var(queue_sizes),
            'peak_queue_size': self.peak_queue_size,
            'time_since_peak_ms': current_time - self.last_peak_time,
            'samples_count': len(queue_sizes),
            'network_jitter_std': self.network_jitter_std,
            'normal_growth_rate': self.normal_growth_rate
        }
    
    def reset(self):
        """Reset detector state"""
        self.queue_history.clear()
        self.slope_history.clear()
        self.recent_slopes.clear()
        self.recent_microburst_slopes.clear()
        self.consecutive_detections = 0
        self.burst_active = False
        self.last_peak_time = 0
        self.peak_queue_size = 0
        self.adaptive_microburst_threshold = self.microburst_threshold
        self.adaptive_trend_threshold = self.trend_threshold
        self.network_jitter_std = 0.0
        self.normal_growth_rate = 0.0
