import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StaffSystem:
    """
    Represents a single staff system (typically 5 lines).
    Stores the y-coordinates of the lines.
    """
    lines: List[int]  # y-coordinates of the 5 lines (sorted ascending)
    
    @property
    def top_line(self) -> int:
        return self.lines[0]
        
    @property
    def bottom_line(self) -> int:
        return self.lines[-1]
        
    @property
    def center_y(self) -> float:
        return sum(self.lines) / len(self.lines)
        
    @property
    def avg_spacing(self) -> float:
        return np.mean(np.diff(self.lines))

class StaffSystemDetector:
    """
    Detects staff systems from a binary mask where staff lines are marked.
    """
    def __init__(self, mask_path: Path):
        self.mask_path = mask_path
        self.mask: Optional[np.ndarray] = None
        self.staff_systems: List[StaffSystem] = []
        
    def load_mask(self) -> None:
        """
        Loads the mask image. Handles both binary (0/1) and visualization (0/127/255) formats.
        Assumes class 1 (or 127) is staff lines.
        """
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {self.mask_path}")
            
        # Load as grayscale
        self.mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"Could not load image: {self.mask_path}")
            
    def detect_staff_lines(self) -> List[StaffSystem]:
        """
        Detects staff lines using horizontal projection and peak finding.
        Returns a list of StaffSystem objects.
        """
        if self.mask is None:
            self.load_mask()
            
        # Extract staff pixels
        # In the mask: 0=bg, 1=staff, 2=symbols (or 127=staff, 255=symbols)
        # We look for either 1 or 127
        staff_pixels = (self.mask == 1) | (self.mask == 127)
        
        # Horizontal projection: sum pixels along each row
        horizontal_projection = np.sum(staff_pixels, axis=1)
        
        # Find peaks (staff lines)
        # height threshold: assume lines have some minimal length
        # distance threshold: lines shouldn't be too close (e.g. < 5 pixels)
        min_height = self.mask.shape[1] * 0.3  # At least 30% of width
        peaks, _ = find_peaks(horizontal_projection, height=min_height, distance=5)
        
        print(f"[StaffDetector] Detected {len(peaks)} candidate lines.")
        
        # Group lines into systems
        self.staff_systems = self._group_lines(peaks)
        print(f"[StaffDetector] Grouped into {len(self.staff_systems)} staff systems.")
        
        return self.staff_systems
        
    def _group_lines(self, peaks: np.ndarray) -> List[StaffSystem]:
        """
        Groups detected peaks into sets of 5 lines.
        """
        if len(peaks) < 5:
            print("[StaffDetector] Warning: Less than 5 lines detected.")
            return []
            
        sorted_peaks = sorted(peaks)
        systems = []
        current_system = []
        
        # Simple clustering based on vertical distance
        # Typical staff line spacing is roughly constant within a page
        # We can infer it or use a heuristic threshold
        
        # First pass: strict grouping by distance
        for i, y in enumerate(sorted_peaks):
            if not current_system:
                current_system.append(y)
                continue
            
            # Distance to previous line in current system
            dist = y - current_system[-1]
            
            # Heuristic: if distance is reasonable (e.g., < 50 pixels), add to group
            # This threshold assumes 300-600 DPI images where line spacing is ~15-40px
            if dist < 60: 
                current_system.append(y)
            else:
                # Gap too large, start new system
                if len(current_system) >= 4: # Allow 4 lines in case one is missed
                     # If we have > 5 lines, maybe it's two systems close together?
                     # For now, just take chunks of 5 if possible
                     self._add_valid_systems(current_system, systems)
                
                current_system = [y]
        
        # Add last group
        if len(current_system) >= 4:
             self._add_valid_systems(current_system, systems)
             
        return systems
    
    def _add_valid_systems(self, lines: List[int], systems_list: List[StaffSystem]):
        """
        Helper to split a group of lines into valid 5-line systems.
        """
        # If we have exactly 5 lines, great
        if len(lines) == 5:
            systems_list.append(StaffSystem(lines))
        # If we have more (e.g. 10), split
        elif len(lines) > 5:
            # Try to find sub-groups or just take first 5
            # Simple approach: chunk by 5
            for i in range(0, len(lines), 5):
                chunk = lines[i:i+5]
                if len(chunk) >= 4: # Allow imperfect systems
                    systems_list.append(StaffSystem(chunk))
        # If we have 4, assume one missed
        elif len(lines) == 4:
            # Estimate missing line? For now just keep as is or warn
            # We'll append it, but logic later might need 5 lines
            # Let's try to extrapolate the missing one if spacing is consistent
            pass # TODO: extrapolated missing line
            systems_list.append(StaffSystem(lines))

