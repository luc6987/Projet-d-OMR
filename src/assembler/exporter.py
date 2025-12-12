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
            
            for assembled_measure in assembled_part.measures:
                m21_measure = music21.stream.Measure(number=assembled_measure.number)
                
                # Add Clef at start of first measure
                if assembled_measure.number == 1:
                    m21_clef = self._convert_clef(assembled_measure.clef)
                    m21_measure.append(m21_clef)
                
                # Add default time signature (4/4) to first measure if not present
                # MuseScore works better with explicit time signatures
                if assembled_measure.number == 1 and not time_sig_added:
                    # Default to 4/4 if we can't determine from the music
                    time_sig = music21.meter.TimeSignature('4/4')
                    m21_measure.append(time_sig)
                    time_sig_added = True
                
                # Add notes
                notes_added = False
                for note_data in assembled_measure.notes:
                    m21_note = self._create_note(note_data)
                    if m21_note:
                        m21_measure.append(m21_note)
                        notes_added = True
                
                # If measure is empty, add a rest to avoid MuseScore import issues
                if not notes_added and assembled_measure.number == 1:
                    # First measure should have at least a clef, so it's OK
                    pass
                elif not notes_added:
                    # Empty measure after first: add a whole rest
                    rest = music21.note.Rest()
                    rest.duration = music21.duration.Duration(4.0)  # Whole rest
                    m21_measure.append(rest)
                        
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
        
    def _create_note(self, data: AssembledNote) -> music21.note.Note:
        if data.pitch == "Unknown":
            # Maybe return a Rest? Or skip?
            return None
            
        try:
            n = music21.note.Note(data.pitch)
            n.duration.quarterLength = data.duration
            
            if data.accidental:
                n.pitch.accidental = music21.pitch.Accidental(data.accidental)
                
            return n
        except Exception as e:
            print(f"[Exporter] Warning: Invalid pitch {data.pitch}: {e}")
            return None

