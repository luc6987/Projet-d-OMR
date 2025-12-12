import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from .staff import StaffSystem
from .symbols import Symbol
from .theory import PitchEngine, ClefType

from .linker import Linker

@dataclass
class AssembledNote:
    """
    Represents a fully assembled note with pitch and duration.
    """
    pitch: str          # e.g. "C4"
    duration: float     # e.g. 1.0 for quarter, 0.5 for eighth
    accidental: Optional[str] = None # e.g. "sharp", "flat"
    original_symbol: Optional[Symbol] = None
    linked_symbols: List[Symbol] = field(default_factory=list)
    
    # Time position (x-coordinate)
    x: float = 0.0

@dataclass
class AssembledMeasure:
    number: int
    notes: List[AssembledNote] = field(default_factory=list)
    clef: ClefType = ClefType.G_CLEF
    
@dataclass
class AssembledPart:
    measures: List[AssembledMeasure] = field(default_factory=list)

class ScoreBuilder:
    """
    Assembles raw symbols and staff lines into a structured musical score.
    """
    def __init__(self, staff_systems: List[StaffSystem], symbols: List[Symbol], image_shape: Tuple[int, int], linker: Optional[Linker] = None):
        self.staff_systems = sorted(staff_systems, key=lambda s: s.center_y)
        self.symbols = symbols
        self.image_shape = image_shape
        self.linker = linker
        
    def build(self, min_symbols_per_part: int = 10, max_parts: int = None) -> Tuple[List[AssembledPart], List[Tuple[Symbol, Symbol]]]:
        """
        Main assembly process.
        Returns a list of Parts and a list of linked symbol pairs for visualization.
        
        Args:
            min_symbols_per_part: Minimum number of symbols required to create a part.
                                  Parts with fewer symbols will be skipped.
            max_parts: Maximum number of parts to create. If None, uses all valid systems.
                      If specified, keeps only the parts with the most symbols.
        
        Returns:
            (parts, linked_pairs) where linked_pairs is a list of (source, target) symbol tuples
        """
        parts = []
        linked_pairs = []  # Collect all linked pairs for visualization
        
        # 1. Assign symbols to staff systems
        system_symbols = self._assign_symbols_to_systems()
        
        # 2. Filter and sort systems by symbol count
        valid_systems = []
        for i, system in enumerate(self.staff_systems):
            syms = system_symbols.get(i, [])
            
            # Skip systems with too few symbols
            if len(syms) < min_symbols_per_part:
                print(f"[Builder] Skipping system {i+1}: only {len(syms)} symbols (minimum: {min_symbols_per_part})")
                continue
            
            valid_systems.append((i, system, syms))
        
        # Sort by symbol count (descending) and limit if needed
        valid_systems.sort(key=lambda x: len(x[2]), reverse=True)
        if max_parts is not None and len(valid_systems) > max_parts:
            print(f"[Builder] Limiting to {max_parts} parts (from {len(valid_systems)} valid systems)")
            valid_systems = valid_systems[:max_parts]
        
        # 3. Process each valid system
        for i, system, syms in valid_systems:
            part, system_links = self._process_system(system, syms)
            parts.append(part)
            linked_pairs.extend(system_links)
            
        return parts, linked_pairs
        
    def _assign_symbols_to_systems(self) -> Dict[int, List[Symbol]]:
        """
        Assigns each symbol to the vertically nearest staff system.
        """
        assignments = {i: [] for i in range(len(self.staff_systems))}
        
        for sym in self.symbols:
            # Find nearest system
            # We compare symbol center_y to system center_y
            cy = sym.center_y
            
            best_idx = -1
            min_dist = float('inf')
            
            for i, system in enumerate(self.staff_systems):
                dist = abs(cy - system.center_y)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            # Threshold check? If symbol is way off any staff, maybe ignore?
            # For now, assign to nearest.
            if best_idx != -1:
                assignments[best_idx].append(sym)
                
        return assignments
        
    def _process_system(self, system: StaffSystem, symbols: List[Symbol]) -> Tuple[AssembledPart, List[Tuple[Symbol, Symbol]]]:
        """
        Processes a single staff system:
        - Sorts symbols by time (x)
        - Detects Clef
        - Groups into measures (barlines)
        - Calculates pitches
        """
        # Sort by x
        symbols.sort(key=lambda s: s.center_x)
        
        part = AssembledPart()
        current_measure = AssembledMeasure(number=1)
        current_clef = ClefType.G_CLEF # Default
        
        # 1. Scan for Clef at the beginning
        # Typically the first few symbols
        for sym in symbols[:5]: # Look at first 5 symbols
            if 'clef' in sym.class_name:
                current_clef = PitchEngine.get_clef_from_name(sym.class_name)
                break
        
        current_measure.clef = current_clef
        
        # Track linked pairs for visualization
        linked_pairs = []
        
        # 2. Iterate through symbols
        i = 0
        while i < len(symbols):
            sym = symbols[i]
            
            # Case: Barline -> New Measure
            if 'barline' in sym.class_name:
                part.measures.append(current_measure)
                new_num = current_measure.number + 1
                current_measure = AssembledMeasure(number=new_num, clef=current_clef)
                i += 1
                continue
            
            # Case: Notehead
            if 'note' in sym.class_name or 'head' in sym.class_name:
                # Find linked symbols (stem, flag, dot, accidental)
                linked_syms = self._find_linked_symbols(sym, symbols)
                
                # Record links for visualization
                for linked_sym in linked_syms:
                    linked_pairs.append((sym, linked_sym))
                
                # Determine Accidental
                accidental = None
                # 1. Try found links
                for l in linked_syms:
                    if self._is_accidental(l):
                        accidental = self._map_accidental(l.class_name)
                        break
                
                # 2. Fallback to simple distance check if no linker or no link found
                if not accidental and i > 0:
                    prev = symbols[i-1]
                    if self._is_accidental(prev) and self._is_close(prev, sym):
                        accidental = self._map_accidental(prev.class_name)
                        
                # Calculate Pitch
                pitch_name = PitchEngine.calculate_pitch(sym.center_y, system.lines, current_clef)
                
                # Calculate Duration
                duration = self._calculate_duration(sym, linked_syms)
                
                note = AssembledNote(
                    pitch=pitch_name,
                    duration=duration,
                    accidental=accidental,
                    original_symbol=sym,
                    linked_symbols=linked_syms,
                    x=sym.center_x
                )
                current_measure.notes.append(note)
                
            # Case: Clef change?
            elif 'clef' in sym.class_name:
                current_clef = PitchEngine.get_clef_from_name(sym.class_name)
                
            i += 1
            
        # Append last measure
        part.measures.append(current_measure)
        
        return part, linked_pairs

    def _find_linked_symbols(self, notehead: Symbol, candidates: List[Symbol]) -> List[Symbol]:
        """
        Finds symbols linked to the notehead using the Linker (MLP).
        """
        if not self.linker:
            return []
            
        linked = []
        # Optimization: only check candidates within a spatial window
        # e.g. +/- 100px x, +/- 300px y
        window_x = 150
        window_y = 400 
        
        for cand in candidates:
            if cand == notehead: continue
            
            # Skip unrelated classes to save compute
            # We care about stems, beams, flags, dots, accidentals
            relevant_types = ['stem', 'beam', 'flag', 'dot', 'accidental', 'sharp', 'flat', 'natural']
            if not any(t in cand.class_name.lower() for t in relevant_types):
                continue

            if abs(notehead.center_x - cand.center_x) > window_x: continue
            if abs(notehead.center_y - cand.center_y) > window_y: continue
            
            prob = self.linker.predict(notehead, cand, self.image_shape)
            # Lower threshold for better recall (0.3 instead of 0.5)
            # The model might output lower probabilities but still be correct
            if prob > 0.3:
                linked.append(cand)
                
        return linked

    def _calculate_duration(self, notehead: Symbol, linked_syms: List[Symbol]) -> float:
        """
        Determines note duration based on notehead type and linked symbols.
        Uses MLP-linked symbols if available, otherwise falls back to geometric proximity.
        """
        # Fallback: if MLP didn't find links, use geometric proximity
        if not linked_syms:
            linked_syms = self._find_nearby_symbols_geometric(notehead)
        
        # 1. Base duration from Notehead
        is_empty = 'empty' in notehead.class_name or 'Half' in notehead.class_name or 'Whole' in notehead.class_name
        
        # Check for Stem
        has_stem = any('stem' in s.class_name for s in linked_syms)
        
        duration = 1.0
        
        if is_empty:
            if has_stem:
                duration = 2.0 # Half note
            else:
                duration = 4.0 # Whole note
        else:
            # Filled notehead
            num_flags_8 = sum(1 for s in linked_syms if '8th' in s.class_name or ('flag' in s.class_name and '16' not in s.class_name))
            num_flags_16 = sum(1 for s in linked_syms if '16th' in s.class_name)
            has_beam = any('beam' in s.class_name for s in linked_syms)
            
            if num_flags_16 > 0:
                duration = 0.25
            elif num_flags_8 > 0:
                duration = 0.5
            elif has_beam:
                duration = 0.5  # Beamed notes are usually 8th
            else:
                duration = 1.0 # Quarter
        
        # Check for Dot
        has_dot = any('dot' in s.class_name.lower() for s in linked_syms)
        if has_dot:
            duration *= 1.5
            
        return duration
    
    def _find_nearby_symbols_geometric(self, notehead: Symbol) -> List[Symbol]:
        """
        Fallback method: find nearby symbols using simple geometric distance.
        Used when MLP linker is unavailable or fails.
        """
        nearby = []
        # Search in all symbols (not just same system, but that's OK for now)
        for sym in self.symbols:
            if sym == notehead:
                continue
            
            # Check if it's a relevant type
            relevant_types = ['stem', 'beam', 'flag', 'dot', 'accidental', 'sharp', 'flat', 'natural']
            if not any(t in sym.class_name.lower() for t in relevant_types):
                continue
            
            # Distance thresholds (more lenient than MLP window)
            dist_x = abs(notehead.center_x - sym.center_x)
            dist_y = abs(notehead.center_y - sym.center_y)
            
            # Stem should be very close (within 50px x, 100px y)
            if 'stem' in sym.class_name:
                if dist_x < 50 and dist_y < 100:
                    nearby.append(sym)
            # Flags and beams should be close horizontally
            elif 'flag' in sym.class_name or 'beam' in sym.class_name:
                if dist_x < 80 and dist_y < 150:
                    nearby.append(sym)
            # Dots should be very close
            elif 'dot' in sym.class_name.lower():
                if dist_x < 30 and dist_y < 30:
                    nearby.append(sym)
            # Accidentals should be close horizontally
            elif any(t in sym.class_name.lower() for t in ['sharp', 'flat', 'natural']):
                if dist_x < 40 and dist_y < 50:
                    nearby.append(sym)
        
        return nearby

    def _is_accidental(self, sym: Symbol) -> bool:
        names = ['sharp', 'flat', 'natural']
        return any(n in sym.class_name for n in names)
        
    def _map_accidental(self, name: str) -> str:
        if 'sharp' in name: return 'sharp'
        if 'flat' in name: return 'flat'
        if 'natural' in name: return 'natural'
        return None
        
    def _is_close(self, s1: Symbol, s2: Symbol, threshold_x: int = 50) -> bool:
        """
        Checks if s1 is close to s2 (mostly x-axis).
        """
        return abs(s1.center_x - s2.center_x) < threshold_x

