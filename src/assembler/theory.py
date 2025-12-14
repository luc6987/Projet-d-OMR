import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional

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
    
    # Order of sharps in key signature (circle of fifths)
    SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    # Order of flats in key signature (circle of fifths, reverse)
    FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
    
    @staticmethod
    def parse_key_signature(key_str: Optional[str]) -> Dict[str, str]:
        """
        Converts key signature string to dictionary mapping note names to accidentals.
        
        Args:
            key_str: Key signature string (e.g., "1#", "3b", "C", "2#")
                    - "C" or None means no accidentals
                    - "1#" means F sharp
                    - "2#" means F and C sharp
                    - "1b" means B flat
                    - "3b" means B, E, A flat
        
        Returns:
            Dictionary mapping note names to accidentals (e.g., {'F': 'sharp', 'C': 'sharp'})
        """
        if not key_str or key_str == "C":
            return {}
        
        result = {}
        
        # Parse sharps (e.g., "1#", "2#", "3#")
        if '#' in key_str:
            try:
                num_sharps = int(key_str.replace('#', ''))
                for i in range(min(num_sharps, len(PitchEngine.SHARP_ORDER))):
                    note = PitchEngine.SHARP_ORDER[i]
                    result[note] = 'sharp'
            except ValueError:
                pass
        
        # Parse flats (e.g., "1b", "2b", "3b")
        elif 'b' in key_str:
            try:
                num_flats = int(key_str.replace('b', ''))
                for i in range(min(num_flats, len(PitchEngine.FLAT_ORDER))):
                    note = PitchEngine.FLAT_ORDER[i]
                    result[note] = 'flat'
            except ValueError:
                pass
        
        return result
    
    @staticmethod
    def calculate_pitch(center_y: float, staff_lines: List[int], clef_type: ClefType = ClefType.G_CLEF,
                       key_signature: Optional[Dict[str, str]] = None) -> str:
        """
        Calculates pitch name (e.g. 'C4', 'F#5') given a Y coordinate.
        Uses precise step calculation: Step = (Y_line - Y_notehead) / (Line_Spacing / 2)
        
        Args:
            center_y: Y-coordinate of the note center (uses center point to handle bbox errors).
            staff_lines: List of 5 y-coordinates of staff lines (sorted ascending/top-to-bottom).
            clef_type: Type of clef for this staff.
            key_signature: Optional dict mapping note names to accidentals (e.g., {'F': 'sharp'}).
                          If provided, applies key signature accidentals to the calculated pitch.
            
        Returns:
            Pitch string (e.g., "C4", "F#4"). Returns "Unknown" if too far.
        """
        if not staff_lines or len(staff_lines) < 2:
            return "Unknown"
        
        # Calculate line spacing more precisely
        # Use median spacing to be robust to outliers
        line_distances = np.diff(staff_lines)
        median_line_dist = np.median(line_distances)
        step_size = median_line_dist / 2.0  # Half-space = 1 step
        
        # Find the closest staff line to minimize error accumulation
        closest_line_idx = -1
        min_dist = float('inf')
        
        for i, line_y in enumerate(staff_lines):
            dist = abs(center_y - line_y)
            if dist < min_dist:
                min_dist = dist
                closest_line_idx = i
                
        # Base steps for the closest line
        # Top line (idx 0) = 0 steps
        # Each subsequent line is 2 steps lower
        base_steps = closest_line_idx * 2
        
        # Calculate offset from the closest line
        closest_line_y = staff_lines[closest_line_idx]
        delta_y_local = center_y - closest_line_y
        steps_local = delta_y_local / step_size
        
        # Total steps from top line
        total_steps = base_steps + steps_local
        
        # Round to nearest half-step for better accuracy
        steps_down_rounded = round(total_steps * 2) / 2.0
        
        # Get base pitch name
        pitch_name = PitchEngine._get_pitch_from_ref(clef_type, int(round(steps_down_rounded)))
        
        # Apply key signature if provided
        if key_signature:
            pitch_name = PitchEngine._apply_key_signature(pitch_name, key_signature)
        
        return pitch_name
    
    @staticmethod
    def _apply_key_signature(pitch_name: str, key_signature: Dict[str, str]) -> str:
        """
        Applies key signature accidentals to a pitch name.
        
        Args:
            pitch_name: Base pitch name (e.g., "F4")
            key_signature: Dict mapping note names to accidentals (e.g., {'F': 'sharp', 'C': 'sharp'})
            
        Returns:
            Pitch name with accidental if needed (e.g., "F#4")
        """
        if not pitch_name or len(pitch_name) < 2:
            return pitch_name
        
        note_name = pitch_name[0]
        octave = pitch_name[1:] if len(pitch_name) > 1 else ""
        
        if note_name in key_signature:
            accidental = key_signature[note_name]
            if accidental == 'sharp':
                return f"{note_name}#{octave}"
            elif accidental == 'flat':
                return f"{note_name}b{octave}"
            elif accidental == 'natural':
                # Natural cancels previous accidental
                return f"{note_name}{octave}"
        
        return pitch_name
        
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

