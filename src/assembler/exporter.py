import music21
from pathlib import Path
from typing import List, Optional

from .builder import AssembledPart, AssembledMeasure, AssembledNote
from .theory import ClefType

class MusicXMLExporter:
    """
    Converts assembled parts into MusicXML format using music21.
    Optimized for MuseScore compatibility.
    """
    
    def export(self, parts: List[AssembledPart], output_path: Path, 
               title: Optional[str] = None, composer: Optional[str] = None) -> None:
        """
        Exports parts to a MusicXML file compatible with MuseScore.
        
        Args:
            parts: List of assembled parts
            output_path: Output file path
            title: Optional title for the score
            composer: Optional composer name
        """
        score = music21.stream.Score()
        
        # Initialize and set metadata for better MuseScore compatibility
        score.metadata = music21.metadata.Metadata()
        if title:
            score.metadata.title = title
        else:
            score.metadata.title = "OMR Reconstructed Score"
            
        if composer:
            score.metadata.composer = composer
        else:
            score.metadata.composer = "OMR System"
        
        for i, assembled_part in enumerate(parts):
            m21_part = music21.stream.Part()
            # Use simple, consistent IDs for MuseScore compatibility
            m21_part.id = f'P{i+1}'
            # Set part name (MuseScore requires non-empty part-name)
            m21_part.partName = f'Part {i+1}'
            
            # Track if we've added time signature to first measure
            time_sig_added = False
            
            prev_measure = None
            for measure_idx, assembled_measure in enumerate(assembled_part.measures):
                m21_measure = music21.stream.Measure(number=assembled_measure.number)
                
                # Add Clef at start of first measure or when it changes
                if measure_idx == 0:
                    m21_clef = self._convert_clef(assembled_measure.clef)
                    m21_measure.append(m21_clef)
                elif prev_measure and prev_measure.clef != assembled_measure.clef:
                    # Clef changed from previous measure
                    m21_clef = self._convert_clef(assembled_measure.clef)
                    m21_measure.append(m21_clef)
                
                # Add key signature at start of first measure or when it changes
                if assembled_measure.key_signature:
                    m21_key = self._convert_key_signature(assembled_measure.key_signature)
                    if m21_key:
                        # Only add if it's the first measure or changed from previous
                        if measure_idx == 0:
                            m21_measure.append(m21_key)
                        elif prev_measure and prev_measure.key_signature != assembled_measure.key_signature:
                            m21_measure.append(m21_key)
                
                # Add time signature at start of first measure or when it changes
                if assembled_measure.time_signature:
                    m21_time = self._convert_time_signature(assembled_measure.time_signature)
                    if m21_time:
                        # Only add if it's the first measure or changed from previous
                        if measure_idx == 0:
                            m21_measure.append(m21_time)
                            time_sig_added = True
                        elif prev_measure and prev_measure.time_signature != assembled_measure.time_signature:
                            m21_measure.append(m21_time)
                elif measure_idx == 0 and not time_sig_added:
                    # Default to 4/4 if we can't determine from the music
                    time_sig = music21.meter.TimeSignature('4/4')
                    m21_measure.append(time_sig)
                    time_sig_added = True
                
                # Mark as implicit if anacrusis
                if assembled_measure.is_implicit:
                    m21_measure.implicit = True
                
                # Add notes
                notes_added = False
                
                # Identify consecutive tuplet groups and assign tuplet numbers
                # This ensures music21 correctly groups tuplet notes
                # Group by consecutive tuplet notes in the measure (not by X coordinate)
                tuplet_groups = []
                current_group = []
                for i, note_data in enumerate(assembled_measure.notes):
                    if note_data.is_tuplet and note_data.time_modification_actual_notes and note_data.time_modification_normal_notes:
                        # Add to current group if it exists, or start new group
                        if len(current_group) > 0:
                            # Check if previous note was also tuplet (consecutive tuplet notes)
                            prev_note = assembled_measure.notes[current_group[-1]]
                            if prev_note.is_tuplet and prev_note.time_modification_actual_notes and prev_note.time_modification_normal_notes:
                                # Same tuplet type (e.g., both triplets) - add to current group
                                if (note_data.time_modification_actual_notes == prev_note.time_modification_actual_notes and
                                    note_data.time_modification_normal_notes == prev_note.time_modification_normal_notes):
                                    current_group.append(i)
                                else:
                                    # Different tuplet type - start new group
                                    tuplet_groups.append(current_group)
                                    current_group = [i]
                            else:
                                # Previous note was not tuplet - start new group
                                tuplet_groups.append(current_group)
                                current_group = [i]
                        else:
                            # Start new group
                            current_group = [i]
                    else:
                        # Non-tuplet note ends current group
                        if len(current_group) > 0:
                            tuplet_groups.append(current_group)
                            current_group = []
                
                # Add final group if exists
                if len(current_group) > 0:
                    tuplet_groups.append(current_group)
                
                # Create mapping from note index to tuplet number
                note_to_tuplet_number = {}
                for group_idx, group in enumerate(tuplet_groups):
                    tuplet_number = group_idx + 1
                    for note_idx in group:
                        note_to_tuplet_number[note_idx] = tuplet_number
                
                # Group notes by X overlap (Simultaneity)
                simul_groups = []
                if assembled_measure.notes:
                    # Sort just in case (though builder usually sorts)
                    sorted_notes_with_idx = sorted(enumerate(assembled_measure.notes), key=lambda p: p[1].x)
                    
                    current_group = [ sorted_notes_with_idx[0] ] # [(idx, note)]
                    
                    for i in range(1, len(sorted_notes_with_idx)):
                        curr_idx, curr_note = sorted_notes_with_idx[i]
                        prev_idx, prev_note = current_group[-1]
                        
                        # Check overlap
                        should_group = False
                        # If both have symbols, check bbox intersection
                        if curr_note.original_symbol and prev_note.original_symbol:
                            l1, r1 = prev_note.original_symbol.x1, prev_note.original_symbol.x2
                            l2, r2 = curr_note.original_symbol.x1, curr_note.original_symbol.x2
                            # Check if intervals overlap (max_left < min_right)
                            if max(l1, l2) < min(r1, r2):
                                should_group = True
                        else:
                            # Fallback to proximity (within 15px)
                            if abs(curr_note.x - prev_note.x) < 15.0:
                                should_group = True
                                
                        if should_group:
                            current_group.append((curr_idx, curr_note))
                        else:
                            simul_groups.append(current_group)
                            current_group = [(curr_idx, curr_note)]
                    simul_groups.append(current_group)
                
                # Process groups
                for group in simul_groups:
                    # group is list of (note_idx, note_data)
                    
                    if len(group) == 1:
                        # Single note
                        note_idx, note_data = group[0]
                        tuplet_number = note_to_tuplet_number.get(note_idx)
                        m21_note = self._create_note(note_data, tuplet_number=tuplet_number)
                        if m21_note:
                            m21_measure.append(m21_note)
                            notes_added = True
                    else:
                        # Chord (multiple notes)
                        chord_notes = []
                        first_valid_note_idx = None
                        
                        for note_idx, note_data in group:
                            tuplet_number = note_to_tuplet_number.get(note_idx)
                            m21_n = self._create_note(note_data, tuplet_number=tuplet_number)
                            if m21_n:
                                chord_notes.append(m21_n)
                                if first_valid_note_idx is None:
                                    first_valid_note_idx = note_idx

                        if chord_notes:
                            if len(chord_notes) == 1:
                                # Only one valid note in group, treat as note
                                m21_measure.append(chord_notes[0])
                            else:
                                # Create Chord
                                m21_chord = music21.chord.Chord(chord_notes)
                                # Explicitly set duration from first note (assume all in chord have same duration)
                                m21_chord.duration = chord_notes[0].duration
                                m21_measure.append(m21_chord)
                            notes_added = True
                
                # If measure is empty, add a rest to avoid MuseScore import issues
                if not notes_added and measure_idx == 0:
                    # First measure should have at least a clef, so it's OK
                    pass
                elif not notes_added:
                    # Empty measure after first: add a whole rest
                    rest = music21.note.Rest()
                    rest.duration = music21.duration.Duration(4.0)  # Whole rest
                    m21_measure.append(rest)
                
                # Update prev_measure for next iteration
                prev_measure = assembled_measure
                        
                m21_part.append(m21_measure)
                
            score.insert(0, m21_part)
            
        # Write to file with MuseScore-compatible settings
        print(f"[Exporter] Writing MusicXML to {output_path}...")
        try:
            # Use music21's write method with format='musicxml'
            # This generates MusicXML 3.1 which is compatible with MuseScore 3.x and 4.x
            score.write('musicxml', fp=str(output_path))
            print("[Exporter] Success.")
            print(f"[Exporter] Generated {len(parts)} parts with MusicXML 3.1 format.")
        except Exception as e:
            print(f"[Exporter] Error writing file: {e}")
            import traceback
            traceback.print_exc()
            
    def _convert_clef(self, clef_type: ClefType) -> music21.clef.Clef:
        if clef_type == ClefType.G_CLEF:
            return music21.clef.TrebleClef()
        elif clef_type == ClefType.F_CLEF:
            return music21.clef.BassClef()
        elif clef_type == ClefType.C_CLEF:
            return music21.clef.AltoClef()
        return music21.clef.TrebleClef()
        
    def _convert_key_signature(self, key_str: str) -> Optional[music21.key.KeySignature]:
        """
        Convert key signature string to music21 KeySignature object.
        
        Args:
            key_str: Key signature string (e.g., "1#", "3b", "C")
            
        Returns:
            music21 KeySignature object, or None if invalid
        """
        try:
            if key_str == "C" or key_str is None:
                return music21.key.KeySignature(0)
            elif key_str.endswith('#'):
                num_sharps = int(key_str[:-1])
                return music21.key.KeySignature(num_sharps)
            elif key_str.endswith('b'):
                num_flats = int(key_str[:-1])
                return music21.key.KeySignature(-num_flats)
            else:
                return None
        except:
            return None
    
    def _convert_time_signature(self, time_str: str) -> Optional[music21.meter.TimeSignature]:
        """
        Convert time signature string to music21 TimeSignature object.
        
        Args:
            time_str: Time signature string (e.g., "4/4", "3/4", "C")
            
        Returns:
            music21 TimeSignature object, or None if invalid
        """
        try:
            if time_str == "C":
                return music21.meter.TimeSignature('4/4')
            elif '/' in time_str:
                return music21.meter.TimeSignature(time_str)
            else:
                return None
        except:
            return None
    
    def _create_note(self, data: AssembledNote, tuplet_number: Optional[int] = None) -> music21.note.Note:
        if data.pitch == "Unknown":
            # Maybe return a Rest? Or skip?
            return None
            
        try:
            n = music21.note.Note(data.pitch)
            n.duration.quarterLength = data.duration
            
            if data.accidental:
                n.pitch.accidental = music21.pitch.Accidental(data.accidental)
            
            # Handle tuplet/time-modification
            if data.is_tuplet and data.time_modification_actual_notes and data.time_modification_normal_notes:
                # Create tuplet: actual_notes in the time of normal_notes
                # For triplets: 3 in the time of 2
                tuplet = music21.duration.Tuplet(
                    data.time_modification_actual_notes,
                    data.time_modification_normal_notes
                )
                # Set tuplet number to ensure proper grouping in music21
                if tuplet_number is not None:
                    tuplet.number = tuplet_number
                n.duration.appendTuplet(tuplet)
                
                # Set bracket display if needed
                if hasattr(data, 'tuplet_bracket') and data.tuplet_bracket:
                    tuplet.bracket = True
                else:
                    tuplet.bracket = False
                
            return n
        except Exception as e:
            print(f"[Exporter] Warning: Invalid pitch {data.pitch}: {e}")
            return None

