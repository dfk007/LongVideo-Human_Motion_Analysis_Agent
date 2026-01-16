"""
Motion Analysis Utilities

This module provides tools for analyzing pose data against biomechanics standards
and converting raw measurements into interpretable coaching insights.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from app.services.pose_estimation import PoseData

logger = logging.getLogger(__name__)


class MotionAnalyzer:
    """
    Analyzes pose data against biomechanics guidelines and extracts
    coaching-relevant insights.
    """
    
    def __init__(self, guidelines_path: Optional[str] = None):
        """
        Initialize the motion analyzer.
        
        Args:
            guidelines_path: Path to biomechanics guidelines JSON file
        """
        if guidelines_path is None:
            # Default to guidelines in services directory
            guidelines_path = Path(__file__).parent / "biomechanics_guidelines.json"
        
        self.guidelines = self._load_guidelines(guidelines_path)
        
    def _load_guidelines(self, path: Path) -> Dict:
        """Load biomechanics guidelines from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load guidelines from {path}: {e}")
            return {"exercises": {}}
    
    def analyze_pose_sequence(
        self,
        pose_sequence: List[PoseData],
        exercise_type: str
    ) -> Dict[str, any]:
        """
        Analyze a sequence of poses for a specific exercise type.
        
        This is the main analysis function that:
        1. Checks poses against safety standards
        2. Identifies violations
        3. Calculates motion statistics
        4. Provides coaching recommendations
        
        Args:
            pose_sequence: List of PoseData objects in temporal order
            exercise_type: Type of exercise (e.g., "squat", "pushup")
            
        Returns:
            Dict containing analysis results with violations, stats, and recommendations
        """
        if not pose_sequence:
            return {"error": "No pose data provided"}
        
        # Get exercise-specific guidelines
        exercise_guidelines = self.guidelines.get("exercises", {}).get(exercise_type.lower())
        if not exercise_guidelines:
            logger.warning(f"No guidelines found for exercise: {exercise_type}")
            exercise_guidelines = {}
        
        # Initialize analysis results
        analysis = {
            "exercise_type": exercise_type,
            "duration": pose_sequence[-1].timestamp - pose_sequence[0].timestamp,
            "frame_count": len(pose_sequence),
            "average_confidence": np.mean([p.confidence for p in pose_sequence]),
            "violations": [],
            "angle_statistics": {},
            "safety_score": 10.0,  # Start at perfect, deduct for violations
            "recommendations": []
        }
        
        # Analyze each pose in sequence
        for pose in pose_sequence:
            violations = self._check_pose_violations(
                pose, 
                exercise_guidelines.get("safety_standards", {})
            )
            
            # Add timestamp and frame info to each violation
            for violation in violations:
                violation['timestamp'] = pose.timestamp
                violation['frame_number'] = pose.frame_number
                analysis['violations'].append(violation)
        
        # Calculate angle statistics across the sequence
        analysis['angle_statistics'] = self._calculate_angle_statistics(pose_sequence)
        
        # Calculate safety score based on violations
        analysis['safety_score'] = self._calculate_safety_score(analysis['violations'])
        
        # Generate recommendations based on violations
        analysis['recommendations'] = self._generate_recommendations(
            analysis['violations'],
            exercise_guidelines.get("common_errors", [])
        )
        
        return analysis
    
    def _check_pose_violations(
        self,
        pose: PoseData,
        safety_standards: Dict
    ) -> List[Dict]:
        """
        Check a single pose against safety standards.
        
        Args:
            pose: PoseData object to check
            safety_standards: Dict of safety rules from guidelines
            
        Returns:
            List of violation dicts
        """
        violations = []
        
        for standard_name, standard in safety_standards.items():
            # Check angle-based standards
            if 'safe_range' in standard:
                violation = self._check_angle_range(
                    pose,
                    standard_name,
                    standard
                )
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _check_angle_range(
        self,
        pose: PoseData,
        standard_name: str,
        standard: Dict
    ) -> Optional[Dict]:
        """
        Check if an angle is within the safe range.
        
        Args:
            pose: PoseData object
            standard_name: Name of the standard (e.g., "knee_angle")
            standard: Standard dict with safe_range, etc.
            
        Returns:
            Violation dict if violated, None otherwise
        """
        # Map standard names to angle keys in pose data
        angle_mapping = {
            'knee_angle': ['left_knee', 'right_knee'],
            'elbow_angle': ['left_elbow', 'right_elbow'],
            'hip_angle': ['left_hip', 'right_hip'],
            'back_angle': ['back_angle']
        }
        
        angle_keys = angle_mapping.get(standard_name, [])
        if not angle_keys:
            return None
        
        safe_range = standard.get('safe_range', [0, 360])
        min_safe, max_safe = safe_range
        
        # Check each relevant angle
        for angle_key in angle_keys:
            if angle_key not in pose.angles:
                continue
            
            angle_value = pose.angles[angle_key]
            
            # Check if angle is outside safe range
            if angle_value < min_safe or angle_value > max_safe:
                return {
                    'standard_name': standard_name,
                    'angle_key': angle_key,
                    'measured_value': round(angle_value, 1),
                    'safe_range': safe_range,
                    'severity': standard.get('violation_risk', 'medium'),
                    'message': standard.get('measurement', '') + f" - Measured: {angle_value:.1f}°, Safe range: {min_safe}-{max_safe}°",
                    'source': standard.get('source', 'Unknown')
                }
        
        return None
    
    def _calculate_angle_statistics(
        self,
        pose_sequence: List[PoseData]
    ) -> Dict[str, Dict]:
        """
        Calculate statistics for each angle across the pose sequence.
        
        Returns dict mapping angle names to statistics (min, max, mean, std)
        """
        stats = {}
        
        # Get all angle names from first pose
        if not pose_sequence:
            return stats
        
        angle_names = pose_sequence[0].angles.keys()
        
        for angle_name in angle_names:
            # Collect all values for this angle
            values = [
                p.angles.get(angle_name, 0) 
                for p in pose_sequence 
                if angle_name in p.angles
            ]
            
            if values:
                stats[angle_name] = {
                    'min': round(float(np.min(values)), 1),
                    'max': round(float(np.max(values)), 1),
                    'mean': round(float(np.mean(values)), 1),
                    'std': round(float(np.std(values)), 1),
                    'range': round(float(np.max(values) - np.min(values)), 1)
                }
        
        return stats
    
    def _calculate_safety_score(self, violations: List[Dict]) -> float:
        """
        Calculate overall safety score (0-10) based on violations.
        
        Deducts points based on:
        - Number of violations
        - Severity of violations (critical > high > medium > low)
        """
        score = 10.0
        
        severity_penalties = {
            'critical': 2.0,
            'high': 1.0,
            'medium': 0.5,
            'low': 0.2
        }
        
        for violation in violations:
            severity = violation.get('severity', 'medium')
            penalty = severity_penalties.get(severity, 0.5)
            score -= penalty
        
        # Ensure score is between 0 and 10
        score = max(0.0, min(10.0, score))
        
        return round(score, 1)
    
    def _generate_recommendations(
        self,
        violations: List[Dict],
        common_errors: List[Dict]
    ) -> List[str]:
        """
        Generate coaching recommendations based on violations.
        
        Args:
            violations: List of detected violations
            common_errors: List of common error patterns from guidelines
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Group violations by standard name
        violation_groups = {}
        for v in violations:
            standard = v['standard_name']
            if standard not in violation_groups:
                violation_groups[standard] = []
            violation_groups[standard].append(v)
        
        # Generate recommendations for each violation type
        for standard_name, viol_list in violation_groups.items():
            # Find matching common error
            matching_error = None
            for error in common_errors:
                if standard_name in error.get('detection', '').lower():
                    matching_error = error
                    break
            
            if matching_error:
                recommendations.append(matching_error.get('correction', ''))
            else:
                # Generic recommendation based on the violation
                avg_value = np.mean([v['measured_value'] for v in viol_list])
                safe_range = viol_list[0]['safe_range']
                recommendations.append(
                    f"Improve {standard_name.replace('_', ' ')}: "
                    f"Current average {avg_value:.1f}°, aim for {safe_range[0]}-{safe_range[1]}°"
                )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def format_analysis_for_llm(self, analysis: Dict) -> str:
        """
        Format analysis results into a clear text summary for LLM consumption.
        
        This creates a structured summary that the LLM can use to generate
        natural language coaching feedback.
        
        Args:
            analysis: Analysis results dict from analyze_pose_sequence
            
        Returns:
            Formatted string summary
        """
        lines = [
            f"MOTION ANALYSIS REPORT",
            f"=" * 50,
            f"Exercise: {analysis['exercise_type']}",
            f"Duration: {analysis['duration']:.1f} seconds",
            f"Frames Analyzed: {analysis['frame_count']}",
            f"Detection Confidence: {analysis['average_confidence']:.1%}",
            f"Safety Score: {analysis['safety_score']}/10",
            "",
            "ANGLE STATISTICS:",
        ]
        
        for angle_name, stats in analysis.get('angle_statistics', {}).items():
            lines.append(
                f"  {angle_name.replace('_', ' ').title()}: "
                f"{stats['min']}° - {stats['max']}° (mean: {stats['mean']}°)"
            )
        
        if analysis.get('violations'):
            lines.append("")
            lines.append("SAFETY VIOLATIONS:")
            
            for v in analysis['violations']:
                lines.append(
                    f"  [{v['severity'].upper()}] at {v['timestamp']:.1f}s: "
                    f"{v['angle_key'].replace('_', ' ')} = {v['measured_value']}° "
                    f"(safe: {v['safe_range'][0]}-{v['safe_range'][1]}°)"
                )
        
        if analysis.get('recommendations'):
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                lines.append(f"  {i}. {rec}")
        
        return "\n".join(lines)


# Convenience function for quick analysis
def analyze_video_motion(
    pose_sequence: List[PoseData],
    exercise_type: str
) -> Dict:
    """
    Quick function to analyze motion from pose sequence.
    
    Args:
        pose_sequence: List of PoseData objects
        exercise_type: Type of exercise
        
    Returns:
        Analysis results dict
    """
    analyzer = MotionAnalyzer()
    return analyzer.analyze_pose_sequence(pose_sequence, exercise_type)