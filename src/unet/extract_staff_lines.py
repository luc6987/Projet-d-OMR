"""
Staff line extraction from U-Net predictions
Extract staff line positions and group them into systems
"""
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StaffLine:
    """Represents a single staff line with its y-coordinate"""
    y: int  # y-coordinate of the line
    
    
@dataclass
class StaffSystem:
    """
    Represents a single staff system (typically 5 lines).
    Stores the y-coordinates of the lines.
    """
    lines: List[int]  # y-coordinates of the lines (sorted ascending)
    
    @property
    def top_line(self) -> int:
        """Top line y-coordinate"""
        return self.lines[0] if self.lines else 0
        
    @property
    def bottom_line(self) -> int:
        """Bottom line y-coordinate"""
        return self.lines[-1] if self.lines else 0
        
    @property
    def center_y(self) -> float:
        """Center y-coordinate of the staff system"""
        if not self.lines:
            return 0.0
        return sum(self.lines) / len(self.lines)
        
    @property
    def avg_spacing(self) -> float:
        """Average spacing between consecutive lines"""
        if len(self.lines) < 2:
            return 0.0
        return float(np.mean(np.diff(self.lines)))


def trim_horizontal_margins(mask: np.ndarray, margin_threshold: float = 0.01) -> Tuple[np.ndarray, int, int]:
    """
    Trim left and right blank margins from mask.
    
    Args:
        mask: Input mask (grayscale, 0-255)
        margin_threshold: Threshold ratio of non-zero pixels to consider as content (default: 0.01)
    
    Returns:
        Tuple of (trimmed_mask, left_offset, right_offset)
    """
    # Vertical projection: sum pixels along each column
    vertical_projection = np.sum(mask > 0, axis=0)
    
    # Find left and right boundaries
    # Left boundary: first column with content
    left_idx = 0
    for i in range(len(vertical_projection)):
        if vertical_projection[i] > mask.shape[0] * margin_threshold:
            left_idx = i
            break
    
    # Right boundary: last column with content
    right_idx = len(vertical_projection) - 1
    for i in range(len(vertical_projection) - 1, -1, -1):
        if vertical_projection[i] > mask.shape[0] * margin_threshold:
            right_idx = i
            break
    
    # Trim the mask
    trimmed_mask = mask[:, left_idx:right_idx + 1]
    
    return trimmed_mask, left_idx, right_idx


def extract_staff_lines_from_mask(
    mask: np.ndarray,
    min_line_length_ratio: float = 0.3,
    min_distance: int = 5,
    staff_class: int = 1,
    trim_margins: bool = True,
    auto_detect_format: bool = True
) -> List[StaffLine]:
    """
    Extract staff line positions from U-Net prediction mask.
    
    Args:
        mask: Prediction mask from U-Net. Can be:
            - Standard format: 0=background, 1=staff, 2=symbols
            - Staff-only format: 0=background, 255=staff
            - Visualization format: 0=background, 127=staff, 255=symbols
        min_line_length_ratio: Minimum ratio of image width for a valid line (default: 0.3)
        min_distance: Minimum distance between peaks in pixels (default: 5)
        staff_class: Class value for staff lines (default: 1). Ignored if auto_detect_format=True
        trim_margins: Whether to trim left/right blank margins (default: True)
        auto_detect_format: Automatically detect mask format (default: True)
    
    Returns:
        List of StaffLine objects with y-coordinates
    """
    if mask.size == 0:
        return []
    
    # Auto-detect format
    unique_vals = np.unique(mask)
    if auto_detect_format:
        if 255 in unique_vals and 127 not in unique_vals and 1 not in unique_vals and 2 not in unique_vals:
            # Staff-only format: 0=background, 255=staff
            staff_pixels = (mask == 255)
        elif 127 in unique_vals or (1 in unique_vals and 2 in unique_vals):
            # Standard or visualization format: extract staff pixels
            # Try standard format first (0, 1, 2)
            if 1 in unique_vals:
                staff_pixels = (mask == 1)
            else:
                # Visualization format (0, 127, 255)
                staff_pixels = (mask == 127)
        else:
            # Fallback: use staff_class parameter
            staff_pixels = (mask == staff_class)
    else:
        # Use provided staff_class
        staff_pixels = (mask == staff_class)
    
    # Trim horizontal margins if requested
    if trim_margins:
        trimmed_mask, left_offset, right_offset = trim_horizontal_margins(
            mask.astype(np.uint8) if staff_pixels.dtype != np.uint8 else mask
        )
        # Re-extract staff pixels from trimmed mask
        if auto_detect_format:
            if 255 in np.unique(trimmed_mask) and 127 not in np.unique(trimmed_mask):
                staff_pixels = (trimmed_mask == 255)
            elif 1 in np.unique(trimmed_mask):
                staff_pixels = (trimmed_mask == 1)
            else:
                staff_pixels = (trimmed_mask == 127)
        else:
            staff_pixels = (trimmed_mask == staff_class)
    else:
        left_offset = 0
        right_offset = 0
    
    # Horizontal projection: sum pixels along each row
    horizontal_projection = np.sum(staff_pixels, axis=1)
    
    # Find peaks (staff lines)
    # Use trimmed width if margins were trimmed
    effective_width = staff_pixels.shape[1] if trim_margins else mask.shape[1]
    min_height = effective_width * min_line_length_ratio
    peaks, properties = find_peaks(
        horizontal_projection, 
        height=min_height, 
        distance=min_distance
    )
    
    # Convert to StaffLine objects
    staff_lines = [StaffLine(y=int(y)) for y in peaks]
    
    return staff_lines


def group_staff_lines(
    staff_lines: List[StaffLine],
    max_gap: int = 60,
    min_lines_per_system: int = 4
) -> List[StaffSystem]:
    """
    Group detected staff lines into systems (typically 5 lines per system).
    
    Args:
        staff_lines: List of StaffLine objects
        max_gap: Maximum gap between lines in the same system (default: 60 pixels)
        min_lines_per_system: Minimum lines required to form a system (default: 4)
    
    Returns:
        List of StaffSystem objects
    """
    if len(staff_lines) < min_lines_per_system:
        return []
    
    # Sort by y-coordinate
    sorted_lines = sorted(staff_lines, key=lambda line: line.y)
    sorted_y = [line.y for line in sorted_lines]
    
    systems = []
    current_system = []
    
    # Group lines by vertical distance
    for i, y in enumerate(sorted_y):
        if not current_system:
            current_system.append(y)
            continue
        
        # Distance to previous line in current system
        dist = y - current_system[-1]
        
        # If distance is reasonable, add to current system
        if dist < max_gap:
            current_system.append(y)
        else:
            # Gap too large, start new system
            if len(current_system) >= min_lines_per_system:
                systems.append(StaffSystem(current_system.copy()))
            current_system = [y]
    
    # Add last group
    if len(current_system) >= min_lines_per_system:
        systems.append(StaffSystem(current_system))
    
    # Post-process: split systems with more than 5 lines
    final_systems = []
    for system in systems:
        if len(system.lines) == 5:
            final_systems.append(system)
        elif len(system.lines) > 5:
            # Split into chunks of 5
            for i in range(0, len(system.lines), 5):
                chunk = system.lines[i:i+5]
                if len(chunk) >= min_lines_per_system:
                    final_systems.append(StaffSystem(chunk))
        else:
            # 4 lines or less, keep as is
            final_systems.append(system)
    
    return final_systems


def extract_and_group_staff_lines(
    mask: np.ndarray,
    min_line_length_ratio: float = 0.3,
    min_distance: int = 5,
    max_gap: int = 60,
    min_lines_per_system: int = 4,
    staff_class: int = 1,
    trim_margins: bool = True,
    auto_detect_format: bool = True
) -> Tuple[List[StaffLine], List[StaffSystem], Optional[Tuple[int, int]]]:
    """
    Extract staff lines from mask and group them into systems.
    
    Args:
        mask: Prediction mask from U-Net. Can be:
            - Standard format: 0=background, 1=staff, 2=symbols
            - Staff-only format: 0=background, 255=staff
            - Visualization format: 0=background, 127=staff, 255=symbols
        min_line_length_ratio: Minimum ratio of image width for a valid line
        min_distance: Minimum distance between peaks in pixels
        max_gap: Maximum gap between lines in the same system
        min_lines_per_system: Minimum lines required to form a system
        staff_class: Class value for staff lines (default: 1). Ignored if auto_detect_format=True
        trim_margins: Whether to trim left/right blank margins (default: True)
        auto_detect_format: Automatically detect mask format (default: True)
    
    Returns:
        Tuple of (staff_lines, staff_systems, trim_info)
        trim_info is (left_offset, right_offset) if trim_margins=True, else None
    """
    # Extract individual lines and get trim info
    trim_info = None
    if trim_margins:
        # Get trim info first
        trimmed_mask, left_offset, right_offset = trim_horizontal_margins(mask)
        trim_info = (left_offset, right_offset)
    
    staff_lines = extract_staff_lines_from_mask(
        mask,
        min_line_length_ratio=min_line_length_ratio,
        min_distance=min_distance,
        staff_class=staff_class,
        trim_margins=trim_margins,
        auto_detect_format=auto_detect_format
    )
    
    # Group into systems
    staff_systems = group_staff_lines(
        staff_lines,
        max_gap=max_gap,
        min_lines_per_system=min_lines_per_system
    )
    
    return staff_lines, staff_systems, trim_info

