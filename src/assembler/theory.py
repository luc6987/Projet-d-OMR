import numpy as np
from enum import Enum
from typing import List, Dict, Tuple

class ClefType(Enum):
    G_CLEF = "g-clef" # Treble
    F_CLEF = "f-clef" # Bass
    C_CLEF = "c-clef" # Alto/Tenor

class PitchEngine:
    """
    Converts geometric vertical position to musical pitch.
    """
    
    # Maps steps relative to TOP line (line index 0) to pitch names
    # Step 0 = Top Line (Line 5)
    # Step 1 = Space above line 5 (wait, usually we count lines from bottom 1..5)
    # Let's use standard convention: Line 1 is bottom. Line 5 is top.
    # BUT in image coordinates, y increases downwards.
    # So Top Line has smallest y. Let's call it "Line 5" (musical) = "Line 0" (array index).
    
    # Treble Clef (G-Clef):
    # Line 5 (Top, index 0): F5
    # Space below (index 0.5?): E5
    # Line 4 (index 1): D5
    # ...
    # Line 1 (Bottom, index 4): E4
    
    # Step definition: 
    # Each line/space is 1 step.
    # Distance between lines is 2 steps.
    
    # Let's define reference pitch for the TOP LINE (Index 0) for each clef
    # Treble: Top line is F5
    # Bass: Top line is A3
    # Alto: Top line is G4 (approx, C is on middle line)
    
    REF_PITCHES = {
        ClefType.G_CLEF: "F5",
        ClefType.F_CLEF: "A3",
        ClefType.C_CLEF: "G4" # Assuming Alto clef where C4 is middle line (Line 3)
                              # Line 3 = C4 -> Line 4 = E4 -> Line 5 = G4
    }

    SCALE = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    
    @staticmethod
    def calculate_pitch(center_y: float, staff_lines: List[int], clef_type: ClefType = ClefType.G_CLEF) -> str:
        """
        Calculates pitch name (e.g. 'C4', 'F#5') given a Y coordinate.
        
        Args:
            center_y: Y-coordinate of the note center.
            staff_lines: List of 5 y-coordinates of staff lines (sorted ascending/top-to-bottom).
            clef_type: Type of clef for this staff.
            
        Returns:
            Pitch string (e.g., "C4"). Returns "Unknown" if too far.
        """
        if not staff_lines:
            return "Unknown"
            
        # Calculate average spacing (half-space = 1 step)
        # Spacing between lines is 2 steps (Line -> Space -> Line)
        avg_line_dist = np.mean(np.diff(staff_lines))
        step_size = avg_line_dist / 2.0
        
        # Reference: Top Line (index 0)
        top_line_y = staff_lines[0]
        
        # Calculate steps from top line
        # Positive delta y (downwards) -> Lower pitch
        delta_y = center_y - top_line_y
        
        # steps = delta_y / step_size
        # Round to nearest integer
        steps_down = round(delta_y / step_size)
        
        return PitchEngine._get_pitch_from_ref(clef_type, steps_down)
        
    @staticmethod
    def _get_pitch_from_ref(clef_type: ClefType, steps_down: int) -> str:
        """
        steps_down: Number of diatonic steps BELOW the top line.
        Negative means above top line.
        """
        ref_pitch = PitchEngine.REF_PITCHES.get(clef_type, "F5")
        
        # Parse ref pitch
        ref_note = ref_pitch[0]
        ref_octave = int(ref_pitch[1])
        
        ref_idx = PitchEngine.SCALE.index(ref_note)
        
        # Calculate target index
        # Moving down 1 step in scale -> index -1
        # We subtract steps_down
        
        # Total "scale indices" (C4=0, D4=1...) logic is easier
        # Let's convert ref to absolute scalar
        # C0 = 0
        abs_ref = ref_octave * 7 + ref_idx
        
        abs_target = abs_ref - steps_down
        
        target_octave = abs_target // 7
        target_note_idx = abs_target % 7
        target_note = PitchEngine.SCALE[target_note_idx]
        
        return f"{target_note}{target_octave}"

    @staticmethod
    def get_clef_from_name(name: str) -> ClefType:
        name = name.lower()
        if "g-clef" in name or "treble" in name:
            return ClefType.G_CLEF
        if "f-clef" in name or "bass" in name:
            return ClefType.F_CLEF
        if "c-clef" in name or "alto" in name:
            return ClefType.C_CLEF
        return ClefType.G_CLEF # Default

