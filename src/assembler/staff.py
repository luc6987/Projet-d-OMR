import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
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
    
    @property
    def vertical_range(self) -> Tuple[int, int]:
        """Returns (top_y, bottom_y) vertical range of the staff system."""
        return (self.top_line, self.bottom_line)
    
    def contains_y(self, y: float, margin: float = 0.0) -> bool:
        """
        Checks if a y-coordinate falls within this staff system's vertical range.
        
        Args:
            y: Y-coordinate to check
            margin: Additional margin (in pixels) to extend the range
            
        Returns:
            True if y is within the range
        """
        top = self.top_line - margin
        bottom = self.bottom_line + margin
        return top <= y <= bottom


@dataclass
class StaffGrouping:
    """
    Represents a group of related staff systems (e.g., piano grand staff).
    For example, a piano score has two systems: treble (top) and bass (bottom).
    """
    systems: List[StaffSystem]  # List of staff systems in this grouping
    
    @property
    def top_system(self) -> StaffSystem:
        """Returns the topmost staff system."""
        return min(self.systems, key=lambda s: s.center_y)
    
    @property
    def bottom_system(self) -> StaffSystem:
        """Returns the bottommost staff system."""
        return max(self.systems, key=lambda s: s.center_y)
    
    @property
    def top_y(self) -> int:
        """Top Y-coordinate of the entire grouping."""
        return self.top_system.top_line
    
    @property
    def bottom_y(self) -> int:
        """Bottom Y-coordinate of the entire grouping."""
        return self.bottom_system.bottom_line
    
    @property
    def center_y(self) -> float:
        """Center Y-coordinate of the entire grouping."""
        return (self.top_y + self.bottom_y) / 2.0
    
    def contains_y(self, y: float, margin: float = 0.0) -> bool:
        """
        Checks if a y-coordinate falls within this grouping's vertical range.
        
        Args:
            y: Y-coordinate to check
            margin: Additional margin (in pixels) to extend the range
            
        Returns:
            True if y is within the range
        """
        top = self.top_y - margin
        bottom = self.bottom_y + margin
        return top <= y <= bottom
    
    def get_system_for_y(self, y: float) -> Optional[StaffSystem]:
        """
        Finds which staff system in this grouping contains the given y-coordinate.
        
        Args:
            y: Y-coordinate to check
            
        Returns:
            The StaffSystem that contains y, or None if none match
        """
        for system in self.systems:
            if system.contains_y(y):
                return system
        
        # If not found, return the nearest system
        best_system = None
        min_dist = float('inf')
        for system in self.systems:
            dist = abs(system.center_y - y)
            if dist < min_dist:
                min_dist = dist
                best_system = system
        
        return best_system

class StaffSystemDetector:
    """
    Detects staff systems from a binary mask where staff lines are marked.
    """
    def __init__(self, mask_path: Path):
        self.mask_path = mask_path
        self.mask: Optional[np.ndarray] = None
        self.staff_systems: List[StaffSystem] = []
        self.staff_groupings: List[StaffGrouping] = []
        
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
    
    def detect_staff_groupings(self, max_gap_ratio: float = 2.5) -> List[StaffGrouping]:
        """
        Detects and groups related staff systems (e.g., piano grand staff).
        
        Groups systems that are close together vertically, which typically indicates
        they belong to the same musical part (e.g., treble and bass clefs in piano music).
        
        Args:
            max_gap_ratio: Maximum ratio of gap between systems to average system height.
                          If gap between two systems is less than max_gap_ratio * avg_height,
                          they are grouped together. Default 2.5 means gap should be less than
                          2.5 times the average system height.
        
        Returns:
            List of StaffGrouping objects
        """
        if not self.staff_systems:
            self.detect_staff_lines()
        
        if len(self.staff_systems) == 0:
            return []
        
        # Sort systems by vertical position
        sorted_systems = sorted(self.staff_systems, key=lambda s: s.center_y)
        
        # Calculate average system height
        avg_height = np.mean([s.bottom_line - s.top_line for s in sorted_systems])
        
        groupings = []
        current_group = [sorted_systems[0]]
        
        for i in range(1, len(sorted_systems)):
            prev_system = sorted_systems[i-1]
            curr_system = sorted_systems[i]
            
            # Calculate gap between systems
            gap = curr_system.top_line - prev_system.bottom_line
            
            # Check if gap is small enough to group together
            if gap < max_gap_ratio * avg_height:
                # Add to current group
                current_group.append(curr_system)
            else:
                # Gap is too large, start new group
                if current_group:
                    groupings.append(StaffGrouping(systems=current_group))
                current_group = [curr_system]
        
        # Add last group
        if current_group:
            groupings.append(StaffGrouping(systems=current_group))
        
        self.staff_groupings = groupings
        print(f"[StaffDetector] Detected {len(groupings)} staff groupings from {len(sorted_systems)} systems.")
        
        return groupings
    
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
            # Handle case with 4 lines (one line might be missing)
            # For now, we create the system as-is. Future improvement could
            # extrapolate the missing line based on consistent spacing
            systems_list.append(StaffSystem(lines))
    
    def overlap_y(self, obj_a_y1: float, obj_a_y2: float, obj_b_y1: float, obj_b_y2: float) -> float:
        """
        Calculate the overlap ratio of two objects on the Y-axis.
        
        Args:
            obj_a_y1, obj_a_y2: Y-coordinates of object A (top, bottom)
            obj_b_y1, obj_b_y2: Y-coordinates of object B (top, bottom)
            
        Returns:
            Overlap ratio (0.0 to 1.0), where 1.0 means complete overlap
        """
        # Calculate intersection
        overlap_top = max(obj_a_y1, obj_b_y1)
        overlap_bottom = min(obj_a_y2, obj_b_y2)
        
        if overlap_top >= overlap_bottom:
            return 0.0
        
        overlap_height = overlap_bottom - overlap_top
        obj_a_height = obj_a_y2 - obj_a_y1
        obj_b_height = obj_b_y2 - obj_b_y1
        
        # Return the ratio of overlap to the smaller object
        min_height = min(obj_a_height, obj_b_height)
        if min_height == 0:
            return 0.0
        
        return overlap_height / min_height
    
    def cluster_systems_by_symbols(self, symbols: List) -> List[List[StaffSystem]]:
        """
        Cluster staff systems into System Groups based on symbols (braces, brackets, measure separators).
        Implements Rule 1 from rule.md: System Clustering.
        
        Args:
            symbols: List of Symbol objects to analyze
            
        Returns:
            List of System Groups, where each group is a list of StaffSystem objects
        """
        if not self.staff_systems:
            self.detect_staff_lines()
        
        if len(self.staff_systems) == 0:
            return []
        
        # Track which systems belong to which group
        system_to_group: Dict[int, int] = {}  # system_index -> group_id
        groups: List[List[StaffSystem]] = []
        next_group_id = 0
        
        # Priority 1: Cluster based on braces/brackets/staff_grouping
        grouping_symbols = []
        for sym in symbols:
            class_name_lower = sym.class_name.lower()
            if any(keyword in class_name_lower for keyword in ['multi-staff_brace', 'multi-staff_bracket', 'staff_grouping']):
                grouping_symbols.append(sym)
        
        for grouping_sym in grouping_symbols:
            # Find all staff systems that overlap with this grouping symbol
            overlapping_systems = []
            for i, system in enumerate(self.staff_systems):
                overlap_ratio = self.overlap_y(
                    grouping_sym.y1, grouping_sym.y2,
                    system.top_line, system.bottom_line
                )
                if overlap_ratio > 0.5:  # Threshold from rule.md
                    overlapping_systems.append(i)
            
            if overlapping_systems:
                # Assign all overlapping systems to the same group
                existing_group_id = None
                for sys_idx in overlapping_systems:
                    if sys_idx in system_to_group:
                        existing_group_id = system_to_group[sys_idx]
                        break
                
                if existing_group_id is None:
                    existing_group_id = next_group_id
                    next_group_id += 1
                    groups.append([])
                
                for sys_idx in overlapping_systems:
                    if sys_idx not in system_to_group:
                        system_to_group[sys_idx] = existing_group_id
                        groups[existing_group_id].append(self.staff_systems[sys_idx])
        
        # Priority 2: Cluster based on measure separators
        measure_separators = []
        for sym in symbols:
            class_name_lower = sym.class_name.lower()
            if 'measure_separator' in class_name_lower:
                measure_separators.append(sym)
        
        for separator in measure_separators:
            # Find systems that the separator connects (top and bottom)
            top_system_idx = None
            bottom_system_idx = None
            
            for i, system in enumerate(self.staff_systems):
                # Check if separator's top touches this system
                if abs(separator.y1 - system.top_line) < 10 or abs(separator.y1 - system.bottom_line) < 10:
                    if top_system_idx is None:
                        top_system_idx = i
                    elif abs(separator.y1 - system.top_line) < abs(separator.y1 - self.staff_systems[top_system_idx].top_line):
                        top_system_idx = i
                
                # Check if separator's bottom touches this system
                if abs(separator.y2 - system.top_line) < 10 or abs(separator.y2 - system.bottom_line) < 10:
                    if bottom_system_idx is None:
                        bottom_system_idx = i
                    elif abs(separator.y2 - system.top_line) < abs(separator.y2 - self.staff_systems[bottom_system_idx].top_line):
                        bottom_system_idx = i
            
            if top_system_idx is not None and bottom_system_idx is not None:
                # Connect all systems between top and bottom
                start_idx = min(top_system_idx, bottom_system_idx)
                end_idx = max(top_system_idx, bottom_system_idx)
                
                # Find or create a group for these systems
                existing_group_id = None
                for sys_idx in range(start_idx, end_idx + 1):
                    if sys_idx in system_to_group:
                        existing_group_id = system_to_group[sys_idx]
                        break
                
                if existing_group_id is None:
                    existing_group_id = next_group_id
                    next_group_id += 1
                    groups.append([])
                
                for sys_idx in range(start_idx, end_idx + 1):
                    if sys_idx not in system_to_group:
                        system_to_group[sys_idx] = existing_group_id
                        groups[existing_group_id].append(self.staff_systems[sys_idx])
        
        # For unassigned systems, each becomes its own group
        unassigned_systems = [i for i in range(len(self.staff_systems)) if i not in system_to_group]
        for sys_idx in unassigned_systems:
            groups.append([self.staff_systems[sys_idx]])
            system_to_group[sys_idx] = next_group_id
            next_group_id += 1
        
        # Sort each group by Y coordinate
        for group in groups:
            group.sort(key=lambda s: s.center_y)
        
        print(f"[StaffDetector] Clustered {len(self.staff_systems)} systems into {len(groups)} system groups.")
        return groups

