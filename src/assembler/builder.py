import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from .staff import StaffSystem, StaffGrouping, StaffSystemDetector
from .symbols import Symbol
from .theory import PitchEngine, ClefType
from .graph import SymbolGraph, EdgeType, NodeType

from .linker import Linker

# Geometric connection thresholds (in pixels)
# These can be made configurable later
STEM_NOTEHEAD_THRESHOLD_X = 50  # Horizontal distance threshold
STEM_NOTEHEAD_THRESHOLD_Y = 100  # Vertical distance threshold
STEM_NOTEHEAD_WIDTH_FACTOR = 1.5  # Stem should be within notehead_width * this factor

FLAG_STEM_THRESHOLD_X = 80  # Horizontal distance for flag-stem connection
FLAG_STEM_THRESHOLD_Y = 150  # Vertical distance for flag-stem connection

BEAM_STEM_THRESHOLD_X = 100  # Horizontal distance for beam-stem connection
BEAM_STEM_THRESHOLD_Y = 50  # Vertical distance for beam-stem connection

DOT_NOTEHEAD_THRESHOLD_X = 30
DOT_NOTEHEAD_THRESHOLD_Y = 30

ACCIDENTAL_NOTEHEAD_THRESHOLD_X = 40
ACCIDENTAL_NOTEHEAD_THRESHOLD_Y = 50

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
    
    # Tuplet information
    is_tuplet: bool = False  # True if this note is part of a tuplet
    tuplet_type: Optional[str] = None  # e.g., "triplet" for triplets
    base_duration: Optional[float] = None  # Base duration before tuplet modification
    time_modification_actual_notes: Optional[int] = None  # Actual notes in tuplet (3 for triplets)
    time_modification_normal_notes: Optional[int] = None  # Normal notes (2 for triplets)
    tuplet_confidence: Optional[str] = None  # "High" or "Low"
    tuplet_rule_triggered: Optional[str] = None  # "Rule1", "Rule2", "Rule3", or "Rule5"
    tuplet_bracket: Optional[bool] = None  # Whether to show bracket in MusicXML

@dataclass
class AssembledMeasure:
    number: int
    notes: List[AssembledNote] = field(default_factory=list)
    clef: ClefType = ClefType.G_CLEF
    key_signature: Optional[str] = None  # e.g., "1#", "3b", "C" (for C Major/A Minor)
    time_signature: Optional[str] = None  # e.g., "4/4", "3/4", "C" (for Common Time)
    is_implicit: bool = False  # True for anacrusis (pickup measure)
    
@dataclass
class AssembledPart:
    measures: List[AssembledMeasure] = field(default_factory=list)

class ScoreBuilder:
    """
    Assembles raw symbols and staff lines into a structured musical score.
    """
    def __init__(self, staff_systems: List[StaffSystem], symbols: List[Symbol], 
                 image_shape: Tuple[int, int], linker: Optional[Linker] = None,
                 staff_groupings: Optional[List[StaffGrouping]] = None,
                 staff_detector: Optional[StaffSystemDetector] = None):
        self.staff_systems = sorted(staff_systems, key=lambda s: s.center_y)
        self.symbols = symbols
        self.image_shape = image_shape
        self.linker = linker
        self.staff_groupings = staff_groupings or []
        self.staff_detector = staff_detector
        # Store system groups and part indices
        self.system_groups: List[List[StaffSystem]] = []
        self.part_indices: Dict[int, int] = {}  # system_index -> part_id
        
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
        
        # Phase 1: System Clustering (Rule 1 from rule.md)
        if self.staff_detector:
            self.system_groups = self.staff_detector.cluster_systems_by_symbols(self.symbols)
        else:
            # Fallback: each system is its own group
            self.system_groups = [[system] for system in self.staff_systems]
        
        # Phase 2: Part Indexing (Rule 2 from rule.md)
        self._assign_part_indices()
        
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
        all_parts = []
        for i, system, syms in valid_systems:
            part_id = self.part_indices.get(i, i + 1)  # Default to 1-indexed
            part, system_links = self._process_system(system, syms, part_id=part_id)
            all_parts.append((part_id, part))  # Store with part_id for grouping
            linked_pairs.extend(system_links)
        
        # Phase 3.5: Merge systems with the same part_id into a single Part
        # Group parts by part_id
        parts_by_id = {}
        for part_id, part in all_parts:
            if part_id not in parts_by_id:
                parts_by_id[part_id] = []
            parts_by_id[part_id].append(part)
        
        # Merge parts with the same part_id
        merged_parts = []
        for part_id in sorted(parts_by_id.keys()):
            part_list = parts_by_id[part_id]
            if len(part_list) == 1:
                # Single system for this part, use as-is
                merged_parts.append(part_list[0])
            else:
                # Multiple systems for this part, merge them
                merged_part = AssembledPart()
                for part in part_list:
                    merged_part.measures.extend(part.measures)
                merged_parts.append(merged_part)
        
        # Phase 4: Global Measure Indexing (Rule 7 from rule.md)
        parts = self._assign_global_measure_numbers(merged_parts)
        
        # Phase 5: Anacrusis Detection (Rule 8 from rule.md)
        self._detect_anacrusis(parts)
            
        return parts, linked_pairs
    
    def _assign_part_indices(self) -> None:
        """
        Implements Rule 2 from rule.md: Part Indexing.
        Assigns Part IDs to each staff system within each System Group.
        Part IDs are assigned from top to bottom (Part_1, Part_2, ...).
        """
        self.part_indices = {}
        
        # Create mapping from system to its index
        system_to_index = {id(system): i for i, system in enumerate(self.staff_systems)}
        
        # Track part counts per system group for consistency checking
        part_counts_per_group = []
        
        for group_idx, system_group in enumerate(self.system_groups):
            # Sort systems in group by Y coordinate (top to bottom)
            sorted_systems = sorted(system_group, key=lambda s: s.center_y)
            
            # Assign Part IDs (1-indexed)
            for part_idx, system in enumerate(sorted_systems, start=1):
                system_id = id(system)
                if system_id in system_to_index:
                    sys_idx = system_to_index[system_id]
                    self.part_indices[sys_idx] = part_idx
            
            part_counts_per_group.append(len(sorted_systems))
        
        # Consistency check: warn if different system groups have different part counts
        if len(set(part_counts_per_group)) > 1:
            print(f"[Builder] Warning: Inconsistent part counts across system groups: {part_counts_per_group}")
            print(f"[Builder] Default behavior: Part IDs assigned sequentially within each group.")
        
    def _assign_symbols_to_systems(self) -> Dict[int, List[Symbol]]:
        """
        Assigns each symbol to the appropriate staff system using StaffGrouping.
        Uses geometric rules: checks if symbol's Y coordinate falls within staff_grouping's range.
        """
        assignments = {i: [] for i in range(len(self.staff_systems))}
        
        # If we have staff groupings, use them for more precise assignment
        if self.staff_groupings:
            return self._assign_with_groupings(assignments)
        
        # Fallback to simple nearest system assignment
        for sym in self.symbols:
            cy = sym.center_y
            
            best_idx = -1
            min_dist = float('inf')
            
            for i, system in enumerate(self.staff_systems):
                dist = abs(cy - system.center_y)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                assignments[best_idx].append(sym)
                
        return assignments
    
    def _assign_with_groupings(self, assignments: Dict[int, List[Symbol]]) -> Dict[int, List[Symbol]]:
        """
        Assigns symbols to systems using StaffGrouping for precise geometric assignment.
        
        Args:
            assignments: Dictionary mapping system index to list of symbols
            
        Returns:
            Updated assignments dictionary
        """
        # Create mapping from system to its index (using id as key since StaffSystem is not hashable)
        system_to_index = {id(system): i for i, system in enumerate(self.staff_systems)}
        
        for sym in self.symbols:
            cy = sym.center_y
            assigned = False
            
            # First, try to find a grouping that contains this symbol's Y coordinate
            for grouping in self.staff_groupings:
                if grouping.contains_y(cy, margin=50.0):  # 50px margin for tolerance
                    # Found a grouping, now find which specific system within it
                    target_system = grouping.get_system_for_y(cy)
                    if target_system:
                        system_id = id(target_system)
                        if system_id in system_to_index:
                            idx = system_to_index[system_id]
                            assignments[idx].append(sym)
                            assigned = True
                            break
            
            # If not assigned to any grouping, fall back to nearest system
            if not assigned:
                best_idx = -1
                min_dist = float('inf')
                
                for i, system in enumerate(self.staff_systems):
                    # Check if symbol's Y is within system's vertical range (with margin)
                    if system.contains_y(cy, margin=50.0):
                        # Symbol is within this system's range
                        dist = abs(cy - system.center_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = i
                
                # If no system contains it, use nearest center
                if best_idx == -1:
                    for i, system in enumerate(self.staff_systems):
                        dist = abs(cy - system.center_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = i
                
                if best_idx != -1:
                    assignments[best_idx].append(sym)
        
        return assignments
        
    def _process_system(self, system: StaffSystem, symbols: List[Symbol], part_id: int = 1) -> Tuple[AssembledPart, List[Tuple[Symbol, Symbol]]]:
        """
        Processes a single staff system:
        - Sorts symbols by time (x)
        - Detects Clef
        - Groups into measures using bucket sort based on barlines
        - Calculates pitches
        """
        # Sort by x
        symbols.sort(key=lambda s: s.center_x)
        
        # 1. Detect Clef at the beginning (initial state)
        current_clef = ClefType.G_CLEF  # Default
        for sym in symbols[:5]:  # Look at first 5 symbols
            if 'clef' in sym.class_name:
                current_clef = PitchEngine.get_clef_from_name(sym.class_name)
                break
        
        # 2. Use bucket sort to group symbols into measures
        measures_data = self._bucket_sort_by_barlines(symbols, current_clef, system)
        
        # 3. Process each measure
        part = AssembledPart()
        linked_pairs = []
        
        # Initialize attribute state (for state persistence)
        attribute_state = {
            'clef': current_clef,
            'key_signature': None,
            'time_signature': None
        }
        
        for measure_num, measure_symbols in enumerate(measures_data, start=1):
            measure = AssembledMeasure(number=measure_num, clef=attribute_state['clef'],
                                     key_signature=attribute_state['key_signature'],
                                     time_signature=attribute_state['time_signature'])
            
            # Phase 3: Attribute Detection (Rules 4, 5, 6 from rule.md)
            # Detect clef (Rule 4)
            detected_clef = self._detect_clef(measure_symbols, system, attribute_state['clef'], measure_num == 1)
            if detected_clef:
                attribute_state['clef'] = detected_clef
                measure.clef = detected_clef
            
            # Detect key signature (Rule 5)
            detected_key = self._detect_key_signature(measure_symbols, system, attribute_state['key_signature'])
            if detected_key:
                attribute_state['key_signature'] = detected_key
                measure.key_signature = detected_key
            
            # Detect time signature (Rule 6)
            detected_time = self._detect_time_signature(measure_symbols, system, attribute_state['time_signature'])
            if detected_time:
                attribute_state['time_signature'] = detected_time
                measure.time_signature = detected_time
            
            # Step 1: Process all notehead-stem connections (Rule 1)
            # Collect all stems and their associated noteheads
            notehead_to_stem = {}  # notehead_id -> stem
            stem_to_noteheads = {}  # stem_id -> list of notehead_ids
            stem_id_to_stem = {}  # stem_id -> stem object
            stem_to_flags = {}  # stem_id -> list of flags (initialized early for virtual stems)
            stem_to_beams = {}  # stem_id -> list of beams (initialized early for virtual stems)
            
            for sym in measure_symbols:
                # Skip barlines (already used for bucketing)
                if 'barline' in sym.class_name:
                    continue
                
                # Clef changes are now handled in attribute detection phase above
                # Skip clef symbols here (they're already processed)
                if 'clef' in sym.class_name:
                    continue
                
                # Process noteheads - Rule 1: connect notehead to stem
                if 'note' in sym.class_name or 'head' in sym.class_name:
                    # For filled noteheads, only connect if stem overlaps
                    # If no overlap, create virtual stem instead
                    is_filled_notehead = 'notehead-full' in sym.class_name.lower()
                    
                    if is_filled_notehead:
                        # For filled noteheads, only accept overlapping stems
                        linked_stem = self._find_overlapping_stem_for_notehead(sym, symbols)
                    else:
                        # For other noteheads, use normal logic (overlap or nearest)
                        linked_stem = self._find_stem_for_notehead(sym, symbols)
                    
                    # Special handling for filled noteheads without overlapping stems
                    if linked_stem is None and is_filled_notehead:
                        # For filled noteheads without overlapping stems, search for beam/flag above/below
                        # and create a virtual stem if found
                        virtual_stem, found_beams, found_flags = self._create_virtual_stem_for_filled_notehead(sym, symbols, system)
                        if virtual_stem is not None:
                            linked_stem = virtual_stem
                            print(f"[Virtual Stem] Created virtual stem for notehead at ({sym.center_x:.1f}, {sym.center_y:.1f}), "
                                  f"found {len(found_beams)} beams, {len(found_flags)} flags")
                            # Store the found beams/flags for this virtual stem
                            # We'll process them in Step 2
                            notehead_id = id(sym)
                            if notehead_id not in notehead_to_stem:
                                stem_id = id(virtual_stem)
                                notehead_to_stem[notehead_id] = virtual_stem
                                stem_id_to_stem[stem_id] = virtual_stem
                                if stem_id not in stem_to_noteheads:
                                    stem_to_noteheads[stem_id] = []
                                stem_to_noteheads[stem_id].append(notehead_id)
                                # Pre-store the found beams/flags
                                if found_beams:
                                    if stem_id not in stem_to_beams:
                                        stem_to_beams[stem_id] = []
                                    stem_to_beams[stem_id].extend(found_beams)
                                if found_flags:
                                    if stem_id not in stem_to_flags:
                                        stem_to_flags[stem_id] = []
                                    stem_to_flags[stem_id].extend(found_flags)
                            continue
                    
                    if linked_stem is not None:
                        notehead_id = id(sym)
                        stem_id = id(linked_stem)
                        notehead_to_stem[notehead_id] = linked_stem
                        stem_id_to_stem[stem_id] = linked_stem
                        if stem_id not in stem_to_noteheads:
                            stem_to_noteheads[stem_id] = []
                        stem_to_noteheads[stem_id].append(notehead_id)
            
            # Step 2: For each unique stem, find flag/beam connections (Rule 2)
            # Note: stem_to_flags and stem_to_beams are already initialized in Step 1
            # for virtual stems that found beams/flags during creation
            
            for stem_id, notehead_ids in stem_to_noteheads.items():
                # Skip if already processed (virtual stems with pre-found beams/flags)
                if stem_id in stem_to_flags or stem_id in stem_to_beams:
                    continue
                
                stem = stem_id_to_stem[stem_id]
                
                # Rule 2: For each stem, find all flags/beams
                flags, beams = self._find_flag_beam_for_stem(stem, symbols)
                if flags:
                    stem_to_flags[stem_id] = flags
                if beams:
                    stem_to_beams[stem_id] = beams
            
            # Step 3: Process noteheads and rests with all linked symbols
            for sym in measure_symbols:
                # Skip barlines (already used for bucketing)
                if 'barline' in sym.class_name:
                    continue
                
                # Handle clef changes
                if 'clef' in sym.class_name:
                    continue  # Already handled
                
                # Process rests (add them as notes with pitch=None)
                is_rest = 'rest' in sym.class_name.lower()
                if is_rest:
                    # Calculate Duration for rest
                    duration = self._calculate_duration(sym, [])
                    
                    note = AssembledNote(
                        pitch=None,  # Rests have no pitch
                        duration=duration,
                        accidental=None,
                        original_symbol=sym,
                        linked_symbols=[],
                        x=sym.center_x
                    )
                    measure.notes.append(note)
                    continue
                
                # Process noteheads
                if 'note' in sym.class_name or 'head' in sym.class_name:
                    # Build complete linked symbols list
                    linked_syms = []
                    notehead_id = id(sym)
                    linked_stem = notehead_to_stem.get(notehead_id)
                    
                    if linked_stem is not None:
                        linked_syms.append(linked_stem)
                        stem_id = id(linked_stem)
                        
                        # Record notehead -> stem link
                        linked_pairs.append((sym, linked_stem))
                        
                        # Add all flags if exist
                        if stem_id in stem_to_flags:
                            for flag in stem_to_flags[stem_id]:
                                linked_syms.append(flag)
                                linked_pairs.append((linked_stem, flag))  # stem -> flag
                        
                        # Add all beams if exist
                        if stem_id in stem_to_beams:
                            for beam in stem_to_beams[stem_id]:
                                linked_syms.append(beam)
                                linked_pairs.append((linked_stem, beam))  # stem -> beam
                    
                    # Find other components (dots, accidentals) - Rule 3
                    other_syms = self._find_other_symbols_for_notehead(sym, symbols, linked_syms)
                    for other_sym in other_syms:
                        linked_syms.append(other_sym)
                        linked_pairs.append((sym, other_sym))  # notehead -> other
                    
                    # Determine Accidental
                    accidental = None
                    # 1. Try found links
                    for l in linked_syms:
                        if self._is_accidental(l):
                            accidental = self._map_accidental(l.class_name)
                            break
                    
                    # 2. Fallback to simple distance check
                    if not accidental:
                        for prev in measure_symbols:
                            if prev == sym:
                                break
                            if self._is_accidental(prev) and self._is_close(prev, sym):
                                accidental = self._map_accidental(prev.class_name)
                                break
                    
                    # Calculate Pitch (use measure's clef, which may have been updated)
                    pitch_name = PitchEngine.calculate_pitch(sym.center_y, system.lines, measure.clef)
                    
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
                    measure.notes.append(note)
                    
                    # Record links for visualization
                    # IMPORTANT: Flags and beams should link to stems, not directly to noteheads
                    # So we record: (notehead, stem), (stem, flag), (stem, beam) instead of direct links
                    linked_stem = None
                    for linked_sym in linked_syms:
                        if 'stem' in linked_sym.class_name.lower():
                            linked_stem = linked_sym
                            linked_pairs.append((sym, linked_sym))  # notehead -> stem
                        elif 'flag' in linked_sym.class_name.lower():
                            # Flag should link to stem, not notehead
                            if linked_stem:
                                linked_pairs.append((linked_stem, linked_sym))  # stem -> flag
                            # If no stem found, skip this flag (shouldn't happen with correct logic)
                        elif 'beam' in linked_sym.class_name.lower():
                            # Beam should link to stem, not notehead
                            if linked_stem:
                                linked_pairs.append((linked_stem, linked_sym))  # stem -> beam
                            # If no stem found, skip this beam (shouldn't happen with correct logic)
                        else:
                            # Other symbols (dot, accidental) link directly to notehead
                            linked_pairs.append((sym, linked_sym))
                    

            
            # Rule 4: Detect triplets - if a beam links to three stems, check for "3" in region a
            # Use self.symbols (all symbols) instead of symbols (current system symbols) 
            # because bracket symbols might be in different systems
            # Pass all previously processed measures so Rule 1 can search across measures
            all_processed_notes = []
            for prev_measure in part.measures:
                all_processed_notes.extend(prev_measure.notes)
            all_processed_notes.extend(measure.notes)  # Include current measure
            # self._detect_triplets(measure, self.symbols, system, all_processed_notes)
            
            part.measures.append(measure)
        
        # Deduplicate linked pairs (in case same link is recorded multiple times)
        # Use a set with tuple of symbol IDs to ensure uniqueness
        unique_pairs = []
        seen_pairs = set()
        for pair in linked_pairs:
            pair_id = (id(pair[0]), id(pair[1]))
            if pair_id not in seen_pairs:
                seen_pairs.add(pair_id)
                unique_pairs.append(pair)
        
        return part, unique_pairs
    
    def _bucket_sort_by_barlines(self, symbols: List[Symbol], default_clef: ClefType, 
                                 system: Optional[StaffSystem] = None) -> List[List[Symbol]]:
        """
        Groups symbols into measures using bucket sort based on barline X coordinates.
        Implements Rule 3 from rule.md: Global Barline Alignment.
        Handles missing barlines and duplicate barlines.
        
        Args:
            symbols: List of symbols sorted by X coordinate
            default_clef: Default clef for the system
            system: StaffSystem object for validating barline positions and geometry
            
        Returns:
            List of measure buckets, each containing symbols for that measure
        """
        # 1. Collect and validate barline symbols (Rule 3: collect barlines)
        barline_symbols = []
        for sym in symbols:
            class_name_lower = sym.class_name.lower()
            is_barline = any(keyword in class_name_lower for keyword in ['barline', 'thin_barline', 'thick_barline', 'repeat'])
            is_measure_separator = 'measure_separator' in class_name_lower
            
            # Check if measure_separator should be treated as barline
            if is_measure_separator:
                # Check if measure_separator spans multiple systems (parts)
                if system is not None and self.staff_systems:
                    spans_multiple_parts = self._check_measure_separator_spans_multiple_parts(sym, system)
                    if not spans_multiple_parts:
                        # measure_separator is within single part, treat as barline
                        is_barline = True
                        print(f"[Builder] measure_separator at x={sym.center_x:.1f} is within single part, treating as barline")
            
            if is_barline:
                # Validate barline geometry and position
                validation_result = self._validate_barline(sym, system)
                if validation_result:
                    barline_symbols.append(sym)
        
        if system:
            print(f"[Builder] Found {len(barline_symbols)} valid barlines (after validation) in system")
        
        # 2. X-axis projection fusion (Rule 3: merge barlines that are close)
        # Calculate threshold: use the larger of (half notehead width) or (15 pixels)
        # This ensures that barlines within 15 pixels are always merged
        notehead_widths = []
        for sym in symbols:
            class_name_lower = sym.class_name.lower()
            if 'note' in class_name_lower or 'head' in class_name_lower:
                notehead_widths.append(sym.width)
        
        if notehead_widths:
            avg_notehead_width = np.mean(notehead_widths)
            threshold = max(avg_notehead_width / 2.0, 15.0)  # At least 15 pixels
        else:
            threshold = 15.0  # Default threshold: 15 pixels
        
        # Merge barlines that are within threshold
        if barline_symbols:
            # Sort by X coordinate
            barline_symbols.sort(key=lambda s: s.center_x)
            
            # Group barlines that are close together
            merged_groups = []
            current_group = [barline_symbols[0]]
            
            for i in range(1, len(barline_symbols)):
                prev_x = current_group[-1].center_x
                curr_x = barline_symbols[i].center_x
                
                if abs(curr_x - prev_x) < threshold:
                    # Merge into current group
                    current_group.append(barline_symbols[i])
                else:
                    # Start new group
                    # Calculate average X for the group
                    avg_x = np.mean([s.center_x for s in current_group])
                    merged_groups.append(avg_x)
                    current_group = [barline_symbols[i]]
            
            # Add last group
            if current_group:
                avg_x = np.mean([s.center_x for s in current_group])
                merged_groups.append(avg_x)
            
            barline_xs = sorted(merged_groups)
        else:
            barline_xs = []
        
        # 2. If no barlines found, infer measure boundaries from symbol spacing
        if not barline_xs:
            return self._infer_measures_from_spacing(symbols)
        
        # 3. Create buckets for each measure
        # First measure: from start to first barline
        # Subsequent measures: between consecutive barlines
        # Last measure: from last barline to end
        
        measures = []
        
        # First measure: symbols before first barline
        first_measure = [s for s in symbols if s.center_x < barline_xs[0]]
        if first_measure:
            measures.append(first_measure)
        
        # Middle measures: between consecutive barlines
        for i in range(len(barline_xs) - 1):
            measure_start = barline_xs[i]
            measure_end = barline_xs[i + 1]
            measure_symbols = [s for s in symbols 
                              if measure_start <= s.center_x < measure_end]
            measures.append(measure_symbols)
        
        # Last measure: symbols after last barline
        last_measure = [s for s in symbols if s.center_x >= barline_xs[-1]]
        if last_measure:
            measures.append(last_measure)
        
        # 4. Filter out empty measures (only containing barlines)
        measures = [m for m in measures if any('barline' not in s.class_name for s in m)]
        
        # Ensure at least one measure exists
        if not measures:
            measures = [symbols]
        
        return measures
    
    def _infer_measures_from_spacing(self, symbols: List[Symbol]) -> List[List[Symbol]]:
        """
        Infers measure boundaries from symbol spacing when no barlines are detected.
        Uses temporal gaps to estimate measure boundaries.
        
        Args:
            symbols: List of symbols sorted by X coordinate
            
        Returns:
            List of measure buckets
        """
        if not symbols:
            return []
        
        # Calculate average symbol spacing
        if len(symbols) < 2:
            return [symbols]
        
        x_coords = [s.center_x for s in symbols]
        gaps = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        threshold = avg_gap * 3  # Large gap indicates measure boundary
        
        measures = []
        current_measure = [symbols[0]]
        
        for i in range(1, len(symbols)):
            gap = x_coords[i] - x_coords[i-1]
            if gap > threshold:
                # Large gap, start new measure
                measures.append(current_measure)
                current_measure = [symbols[i]]
            else:
                current_measure.append(symbols[i])
        
        if current_measure:
            measures.append(current_measure)
        
        return measures if measures else [symbols]
    
    def _validate_barline(self, barline: Symbol, system: Optional[StaffSystem] = None) -> bool:
        """
        Validates if a detected barline symbol is actually a valid barline.
        Checks geometry (height, aspect ratio) and position (within staff system).
        
        Args:
            barline: The barline symbol to validate
            system: StaffSystem object for position validation
            
        Returns:
            True if the barline is valid, False otherwise
        """
        # 1. Check aspect ratio: barlines should be very narrow (width << height)
        aspect_ratio = barline.width / max(barline.height, 1.0)  # Avoid division by zero
        if aspect_ratio > 0.3:  # Barlines should have width/height < 0.3
            return False
        
        # 2. Check minimum height: barlines should be reasonably tall
        if barline.height < 20:  # Minimum height threshold (pixels)
            return False
        
        # 3. If system is provided, check if barline is within staff system range
        if system is not None:
            # Barline should span a significant portion of the staff system
            # Check if barline overlaps with staff system vertically
            barline_top = barline.y1
            barline_bottom = barline.y2
            system_top = system.top_line
            system_bottom = system.bottom_line
            
            # Calculate overlap
            overlap_top = max(barline_top, system_top)
            overlap_bottom = min(barline_bottom, system_bottom)
            overlap_height = max(0, overlap_bottom - overlap_top)
            
            # Barline should overlap at least 50% of staff system height
            system_height = system_bottom - system_top
            min_overlap_ratio = 0.5
            
            if system_height > 0:
                overlap_ratio = overlap_height / system_height
                if overlap_ratio < min_overlap_ratio:
                    return False
            
            # Also check if barline center is roughly within system range (with margin)
            margin = system.avg_spacing * 2
            center_in_range = system.contains_y(barline.center_y, margin=margin)
            if not center_in_range:
                return False
        
        return True
    
    def _check_measure_separator_spans_multiple_parts(self, separator: Symbol, current_system: StaffSystem) -> bool:
        """
        Checks if a measure_separator spans multiple parts (systems).
        A measure_separator should only be treated as a separator if it connects
        different systems. If it's within a single system, it should be treated as a barline.
        
        Args:
            separator: The measure_separator symbol to check
            current_system: The current system being processed
            
        Returns:
            True if separator spans multiple parts, False if it's within a single part
        """
        if not self.staff_systems:
            return False
        
        separator_top = separator.y1
        separator_bottom = separator.y2
        
        # Find which systems the separator overlaps with
        overlapping_systems = []
        for i, system in enumerate(self.staff_systems):
            # Check if separator overlaps with this system
            system_top = system.top_line
            system_bottom = system.bottom_line
            
            # Calculate overlap
            overlap_top = max(separator_top, system_top)
            overlap_bottom = min(separator_bottom, system_bottom)
            overlap_height = max(0, overlap_bottom - overlap_top)
            
            # If overlap is significant (at least 30% of system height), consider it overlapping
            system_height = system_bottom - system_top
            if system_height > 0 and overlap_height / system_height > 0.3:
                overlapping_systems.append(i)
        
        # If separator overlaps with more than one system, it spans multiple parts
        return len(overlapping_systems) > 1

    def _find_stem_for_notehead(self, notehead: Symbol, candidates: List[Symbol]) -> Optional[Symbol]:
        """
        Rule 1: For each notehead, if it overlaps with a stem, connect it.
        If no overlap, find the nearest stem.
        
        Returns:
            The best stem to connect to the notehead, or None if no stem found
        """
        best_stem = None
        best_stem_score = float('inf')
        
        overlapping_stems = []
        non_overlapping_stems = []
        
        for cand in candidates:
            if cand == notehead:
                continue
            
            cand_name_lower = cand.class_name.lower()
            
            # Stem-Notehead connection
            if 'stem' in cand_name_lower:
                # Check if they overlap
                if self._check_bbox_overlap(notehead, cand):
                    overlapping_stems.append(cand)
                else:
                    # Calculate distance for non-overlapping stems
                    dist_sq = self._calculate_bbox_distance(notehead, cand)
                    non_overlapping_stems.append((cand, dist_sq))
        
        # Priority: overlapping stems first, then nearest non-overlapping
        if overlapping_stems:
            # If multiple overlapping stems, choose the one with minimum distance
            for stem in overlapping_stems:
                dist_sq = self._calculate_bbox_distance(notehead, stem)
                if dist_sq < best_stem_score:
                    best_stem_score = dist_sq
                    best_stem = stem
        else:
            # No overlap, find the nearest stem
            if non_overlapping_stems:
                non_overlapping_stems.sort(key=lambda x: x[1])
                best_stem = non_overlapping_stems[0][0]
        
        return best_stem
    
    def _find_overlapping_stem_for_notehead(self, notehead: Symbol, candidates: List[Symbol]) -> Optional[Symbol]:
        """
        For filled noteheads, only find stems that overlap with the notehead.
        Do not connect to non-overlapping stems (they should create virtual stems instead).
        
        Returns:
            The overlapping stem, or None if no overlapping stem found
        """
        best_stem = None
        best_stem_score = float('inf')
        
        overlapping_stems = []
        
        for cand in candidates:
            if cand == notehead:
                continue
            
            cand_name_lower = cand.class_name.lower()
            
            # Stem-Notehead connection - only check overlap
            if 'stem' in cand_name_lower:
                # Check if they overlap
                if self._check_bbox_overlap(notehead, cand):
                    overlapping_stems.append(cand)
        
        # If multiple overlapping stems, choose the one with minimum distance
        if overlapping_stems:
            for stem in overlapping_stems:
                dist_sq = self._calculate_bbox_distance(notehead, stem)
                if dist_sq < best_stem_score:
                    best_stem_score = dist_sq
                    best_stem = stem
        
        return best_stem
    
    def _create_virtual_stem_for_filled_notehead(self, notehead: Symbol, all_symbols: List[Symbol], 
                                                  system: StaffSystem) -> Tuple[Optional[Symbol], List[Symbol], List[Symbol]]:
        """
        For filled noteheads without stems, search for beam/flag above/below.
        If found, create a virtual stem and connect to all found beams/flags.
        If not found, still create a virtual stem.
        
        Search range: within half the distance to the next staff system.
        
        Args:
            notehead: The filled notehead without a stem
            all_symbols: All symbols in the system
            system: The staff system containing this notehead
            
        Returns:
            Tuple of (virtual_stem, found_beams, found_flags)
            virtual_stem: A virtual stem Symbol, or None if creation fails
            found_beams: List of beams found above/below
            found_flags: List of flags found above/below
        """
        # Calculate search range: half the distance to next staff system
        # Find the next staff system (above or below)
        notehead_y = notehead.center_y
        
        # Find distances to adjacent staff systems
        distances_to_other_systems = []
        for other_system in self.staff_systems:
            if other_system == system:
                continue
            dist = abs(other_system.center_y - system.center_y)
            distances_to_other_systems.append(dist)
        
        # Use half of the minimum distance to another system, or default to 2 * avg_spacing
        if distances_to_other_systems:
            max_search_range = min(distances_to_other_systems) / 2.0
        else:
            max_search_range = system.avg_spacing * 2.0
        
        # Search for beams and flags above and below the notehead
        # Search in positive direction (down) and negative direction (up)
        found_beams = []
        found_flags = []
        
        for cand in all_symbols:
            if cand == notehead:
                continue
            
            cand_name_lower = cand.class_name.lower()
            
            # Check if candidate is in the vertical search range (above or below)
            vertical_dist = abs(cand.center_y - notehead_y)
            if vertical_dist > max_search_range:
                continue
            
            # Check if candidate is roughly aligned horizontally (within notehead width * 2)
            horizontal_dist = abs(cand.center_x - notehead.center_x)
            if horizontal_dist > notehead.width * 2:
                continue
            
            # Check beams - must be directly above (negative y) or below (positive y)
            if 'beam' in cand_name_lower:
                # Beam should be directly above or below, not overlapping
                if abs(cand.center_y - notehead_y) > notehead.height / 2:
                    found_beams.append(cand)
            
            # Check flags - must be directly above (negative y) or below (positive y)
            if 'flag' in cand_name_lower:
                # Flag should be directly above or below, not overlapping
                if abs(cand.center_y - notehead_y) > notehead.height / 2:
                    found_flags.append(cand)
        
        # Create virtual stem
        # Position: vertically aligned with notehead, extending up or down based on found beams/flags
        stem_x = notehead.center_x
        stem_width = 3.0  # Typical stem width in pixels
        
        # Determine stem direction based on notehead position relative to middle line
        # Rule: If notehead is at or below the middle line, stem extends upward (flag up)
        #       If notehead is above the middle line, stem extends downward (flag down)
        middle_line_y = system.lines[2]  # Third line (0-indexed, so index 2 is the middle)
        notehead_below_middle = notehead.center_y >= middle_line_y
        
        # Determine stem direction and length based on found beams/flags and notehead position
        if found_beams or found_flags:
            # If beams/flags found, extend stem to reach them
            all_targets = found_beams + found_flags
            min_y = min([t.y1 for t in all_targets] + [notehead.y1])
            max_y = max([t.y2 for t in all_targets] + [notehead.y2])
            
            # Extend a bit beyond to ensure connection
            stem_y1 = min_y - 5
            stem_y2 = max_y + 5
        else:
            # If no beams/flags, create a default-length stem based on notehead position
            # Typical stem length is about 3-4 staff line spacings
            stem_length = system.avg_spacing * 3.5
            
            if notehead_below_middle:
                # Notehead at or below middle line: stem extends upward (flag up)
                stem_y1 = notehead.center_y - stem_length
                stem_y2 = notehead.center_y
            else:
                # Notehead above middle line: stem extends downward (flag down)
                stem_y1 = notehead.center_y
                stem_y2 = notehead.center_y + stem_length
        
        # Create virtual stem Symbol
        virtual_stem = Symbol(
            class_name="stem",
            confidence=0.5,  # Virtual stem has lower confidence
            bbox=[stem_x - stem_width/2, stem_y1, stem_x + stem_width/2, stem_y2]
        )
        
        return virtual_stem, found_beams, found_flags
    
    def _find_flag_beam_for_stem(self, stem: Symbol, candidates: List[Symbol]) -> Tuple[List[Symbol], List[Symbol]]:
        """
        Rule 2: For each stem, if it overlaps with flag/beam, connect it.
        If no overlap, find the nearest flag/beam.
        Keep finding all flags/beams until no more can be found.
        
        Additional rule: If no flag or beam is within one stem width distance, don't add any links.
        
        Returns:
            (flags_list, beams_list) tuple - lists of all connected flags and beams
        """
        found_flags = []
        found_beams = []
        
        # Get stem width for proximity check
        stem_width = stem.width
        stem_width_sq = stem_width * stem_width  # Use squared distance for comparison
        
        # Collect all candidate flags and beams
        overlapping_flags = []
        non_overlapping_flags = []
        overlapping_beams = []
        non_overlapping_beams = []
        
        for cand in candidates:
            if cand == stem:
                continue
            
            cand_name_lower = cand.class_name.lower()
            
            # Check flags
            if 'flag' in cand_name_lower:
                if self._check_bbox_overlap(cand, stem):
                    overlapping_flags.append(cand)
                else:
                    dist_sq = self._calculate_bbox_distance(cand, stem)
                    # Only consider flags within one stem width
                    if dist_sq <= stem_width_sq:
                        non_overlapping_flags.append((cand, dist_sq))
            
            # Check beams
            if 'beam' in cand_name_lower:
                if self._check_bbox_overlap(cand, stem):
                    overlapping_beams.append(cand)
                else:
                    dist_sq = self._calculate_bbox_distance(cand, stem)
                    # Only consider beams within one stem width
                    if dist_sq <= stem_width_sq:
                        non_overlapping_beams.append((cand, dist_sq))
        
        # Check if there's any flag or beam within one stem width
        has_flag_within_width = len(overlapping_flags) > 0 or len(non_overlapping_flags) > 0
        has_beam_within_width = len(overlapping_beams) > 0 or len(non_overlapping_beams) > 0
        
        # If no flag or beam is within one stem width, don't add any links
        if not has_flag_within_width and not has_beam_within_width:
            return [], []
        
        # Add all overlapping flags first (they have highest priority)
        found_flags.extend(overlapping_flags)
        
        # Then add non-overlapping flags, sorted by distance
        if non_overlapping_flags:
            non_overlapping_flags.sort(key=lambda x: x[1])
            found_flags.extend([flag for flag, _ in non_overlapping_flags])
        
        # Add all overlapping beams first (they have highest priority)
        found_beams.extend(overlapping_beams)
        
        # Then add non-overlapping beams, sorted by distance
        if non_overlapping_beams:
            non_overlapping_beams.sort(key=lambda x: x[1])
            found_beams.extend([beam for beam, _ in non_overlapping_beams])
        
        return found_flags, found_beams
    
    def _find_other_symbols_for_notehead(self, notehead: Symbol, candidates: List[Symbol], 
                                         existing_links: List[Symbol]) -> List[Symbol]:
        """
        Find other symbols linked to notehead (dots, accidentals).
        Rule 3: For each notehead, search for dot to the right.
        
        Args:
            notehead: The notehead symbol
            candidates: All candidate symbols
            existing_links: Already linked symbols (to avoid duplicates)
            
        Returns:
            List of other linked symbols
        """
        linked = []
        existing_ids = {id(link) for link in existing_links}
        
        for cand in candidates:
            if cand == notehead or id(cand) in existing_ids:
                continue
            
            cand_name_lower = cand.class_name.lower()
            
            # Dot-Notehead connection (Rule 3: search to the right)
            if 'dot' in cand_name_lower:
                if self._check_dot_notehead_connection(notehead, cand):
                    linked.append(cand)
                    continue
            
            # Accidental-Notehead connection
            if any(t in cand_name_lower for t in ['sharp', 'flat', 'natural', 'accidental']):
                if self._check_accidental_notehead_connection(notehead, cand):
                    linked.append(cand)
                    continue
        
        return linked
    
    def _find_linked_symbols(self, notehead: Symbol, candidates: List[Symbol]) -> List[Symbol]:
        """
        Finds symbols linked to the notehead using geometric rules only.
        Uses Rule-First approach: pure geometric logic, no MLP.
        This method is kept for backward compatibility but is now deprecated.
        Use _find_stem_for_notehead, _find_flag_beam_for_stem, _find_other_symbols_for_notehead instead.
        """
        # Use geometric hard logic only (no MLP refinement)
        linked = self._find_linked_symbols_geometric(notehead, candidates)
        
        return linked
    
    def _find_linked_symbols_geometric(self, notehead: Symbol, candidates: List[Symbol]) -> List[Symbol]:
        """
        Stage 1: Geometric hard logic for component assembly.
        Uses IoU, distance thresholds, and collision detection.
        Ensures each notehead connects to at most one stem.
        
        Args:
            notehead: The notehead symbol to find links for
            candidates: List of candidate symbols to check
            
        Returns:
            List of symbols linked to the notehead
        """
        linked = []
        best_stem = None
        best_stem_score = float('inf')
        
        # Rule 1: For each notehead, if it overlaps with a stem, connect it.
        # If no overlap, find the nearest stem.
        overlapping_stems = []
        non_overlapping_stems = []
        
        for cand in candidates:
            if cand == notehead:
                continue
            
            cand_name_lower = cand.class_name.lower()
            
            # Stem-Notehead connection
            if 'stem' in cand_name_lower:
                # Check if they overlap
                if self._check_bbox_overlap(notehead, cand):
                    overlapping_stems.append(cand)
                else:
                    # Calculate distance for non-overlapping stems
                    dist_sq = self._calculate_bbox_distance(notehead, cand)
                    non_overlapping_stems.append((cand, dist_sq))
        
        # Priority: overlapping stems first, then nearest non-overlapping
        if overlapping_stems:
            # If multiple overlapping stems, choose the one with minimum distance
            for stem in overlapping_stems:
                dist_sq = self._calculate_bbox_distance(notehead, stem)
                if dist_sq < best_stem_score:
                    best_stem_score = dist_sq
                    best_stem = stem
        else:
            # No overlap, find the nearest stem
            if non_overlapping_stems:
                non_overlapping_stems.sort(key=lambda x: x[1])
                best_stem = non_overlapping_stems[0][0]
        
        # Add the best stem if found
        if best_stem is not None:
            linked.append(best_stem)
        
        # Rule 2: For each stem, if it overlaps with flag/beam, connect it.
        # If no overlap, find the nearest flag/beam.
        # This should be done for the stem we just linked, searching in all candidates
        if best_stem is not None:
            # Find flags and beams that can connect to this stem
            best_flag = None
            best_flag_distance = float('inf')
            overlapping_flags = []
            non_overlapping_flags = []
            
            best_beam = None
            best_beam_distance = float('inf')
            overlapping_beams = []
            non_overlapping_beams = []
            
            for cand in candidates:
                if cand == notehead or cand is best_stem:
                    continue
                
                cand_name_lower = cand.class_name.lower()
                
                # Check flags
                if 'flag' in cand_name_lower:
                    if self._check_bbox_overlap(cand, best_stem):
                        overlapping_flags.append(cand)
                    else:
                        dist_sq = self._calculate_bbox_distance(cand, best_stem)
                        non_overlapping_flags.append((cand, dist_sq))
                
                # Check beams
                if 'beam' in cand_name_lower:
                    if self._check_bbox_overlap(cand, best_stem):
                        overlapping_beams.append(cand)
                    else:
                        dist_sq = self._calculate_bbox_distance(cand, best_stem)
                        non_overlapping_beams.append((cand, dist_sq))
            
            # Priority: overlapping flags first, then nearest non-overlapping
            if overlapping_flags:
                for flag in overlapping_flags:
                    dist_sq = self._calculate_bbox_distance(flag, best_stem)
                    if dist_sq < best_flag_distance:
                        best_flag_distance = dist_sq
                        best_flag = flag
            else:
                if non_overlapping_flags:
                    non_overlapping_flags.sort(key=lambda x: x[1])
                    best_flag = non_overlapping_flags[0][0]
            
            # Priority: overlapping beams first, then nearest non-overlapping
            if overlapping_beams:
                for beam in overlapping_beams:
                    dist_sq = self._calculate_bbox_distance(beam, best_stem)
                    if dist_sq < best_beam_distance:
                        best_beam_distance = dist_sq
                        best_beam = beam
            else:
                if non_overlapping_beams:
                    non_overlapping_beams.sort(key=lambda x: x[1])
                    best_beam = non_overlapping_beams[0][0]
            
            # Add flag and beam if found
            if best_flag is not None:
                linked.append(best_flag)
            if best_beam is not None:
                linked.append(best_beam)
        
        # Third pass: find other components (dots, accidentals)
        for cand in candidates:
            if cand == notehead or cand is best_stem or cand in linked:
                continue
            
            cand_name_lower = cand.class_name.lower()
            
            # Dot-Notehead connection
            if 'dot' in cand_name_lower:
                if self._check_dot_notehead_connection(notehead, cand):
                    linked.append(cand)
                    continue
            
            # Accidental-Notehead connection
            if any(t in cand_name_lower for t in ['sharp', 'flat', 'natural', 'accidental']):
                if self._check_accidental_notehead_connection(notehead, cand):
                    linked.append(cand)
                    continue
        
        return linked
    
    def _check_bbox_overlap(self, bbox1: Symbol, bbox2: Symbol) -> bool:
        """
        Check if two bounding boxes overlap (have intersection).
        According to rule.md: check if symbols have overlap.
        
        Args:
            bbox1: First symbol
            bbox2: Second symbol
            
        Returns:
            True if bounding boxes overlap
        """
        # Check horizontal overlap
        if bbox1.x2 < bbox2.x1 or bbox1.x1 > bbox2.x2:
            return False
        
        # Check vertical overlap
        if bbox1.y2 < bbox2.y1 or bbox1.y1 > bbox2.y2:
            return False
        
        return True
    
    def _calculate_bbox_distance(self, bbox1: Symbol, bbox2: Symbol) -> float:
        """
        Calculate distance between two bounding boxes.
        According to rule.md: d = sup_{a in s_i, b in t_i} (a-b)^2
        For practical purposes, we use the minimum distance between any two points.
        
        Args:
            bbox1: First symbol (e.g., stem)
            bbox2: Second symbol (e.g., notehead)
            
        Returns:
            Squared distance between the two bounding boxes
        """
        # Calculate minimum distance between two rectangles
        # If they overlap, distance is 0
        if self._check_bbox_overlap(bbox1, bbox2):
            return 0.0
        
        # Calculate horizontal distance
        if bbox1.x2 < bbox2.x1:
            dx = bbox2.x1 - bbox1.x2
        elif bbox2.x2 < bbox1.x1:
            dx = bbox1.x1 - bbox2.x2
        else:
            dx = 0
        
        # Calculate vertical distance
        if bbox1.y2 < bbox2.y1:
            dy = bbox2.y1 - bbox1.y2
        elif bbox2.y2 < bbox1.y1:
            dy = bbox1.y1 - bbox2.y2
        else:
            dy = 0
        
        # Return squared distance
        return dx * dx + dy * dy
    
    def _check_stem_notehead_connection(self, notehead: Symbol, stem: Symbol) -> bool:
        """
        Checks if stem connects to notehead using geometric rules.
        According to rule.md Rule 1: Check if stem and notehead overlap, 
        if not, find the nearest stem.
        """
        # First check if they overlap
        if self._check_bbox_overlap(notehead, stem):
            return True
        
        # If no overlap, check if within reasonable distance threshold
        # This allows finding the nearest stem when no overlap exists
        dist_sq = self._calculate_bbox_distance(stem, notehead)
        max_dist_sq = (STEM_NOTEHEAD_THRESHOLD_X ** 2) + (STEM_NOTEHEAD_THRESHOLD_Y ** 2)
        
        return dist_sq <= max_dist_sq
    
    def _check_flag_stem_connection(self, flag: Symbol, stem: Symbol) -> bool:
        """
        Checks if flag connects to stem.
        According to rule.md Rule 2: Check if flag and stem overlap,
        if not, find the nearest flag/beam.
        """
        # First check if they overlap
        if self._check_bbox_overlap(flag, stem):
            return True
        
        # If no overlap, check if within reasonable distance threshold
        dist_sq = self._calculate_bbox_distance(flag, stem)
        max_dist_sq = (FLAG_STEM_THRESHOLD_X ** 2) + (FLAG_STEM_THRESHOLD_Y ** 2)
        
        return dist_sq <= max_dist_sq
    
    def _check_beam_stem_connection(self, beam: Symbol, stem: Symbol) -> bool:
        """
        Checks if beam connects to stem.
        According to rule.md Rule 2: Check if beam and stem overlap,
        if not, find the nearest flag/beam.
        """
        # First check if they overlap
        if self._check_bbox_overlap(beam, stem):
            return True
        
        # If no overlap, check if within reasonable distance threshold
        dist_sq = self._calculate_bbox_distance(beam, stem)
        max_dist_sq = (BEAM_STEM_THRESHOLD_X ** 2) + (BEAM_STEM_THRESHOLD_Y ** 2)
        
        return dist_sq <= max_dist_sq
    
    def _check_dot_notehead_connection(self, notehead: Symbol, dot: Symbol) -> bool:
        """
        Checks if dot is to the right of notehead (augmentation dot).
        According to rule.md Rule 3: For each notehead, search for dot to the right.
        """
        # Dot must be to the right of notehead (positive x direction)
        if dot.center_x <= notehead.center_x:
            return False
        
        # Check if within reasonable distance
        dist_x = dot.center_x - notehead.center_x
        dist_y = abs(notehead.center_y - dot.center_y)
        
        return dist_x < DOT_NOTEHEAD_THRESHOLD_X and dist_y < DOT_NOTEHEAD_THRESHOLD_Y
    
    def _check_accidental_notehead_connection(self, notehead: Symbol, accidental: Symbol) -> bool:
        """Checks if accidental is near notehead (usually to the left)."""
        dist_x = abs(notehead.center_x - accidental.center_x)
        dist_y = abs(notehead.center_y - accidental.center_y)
        
        if dist_x > ACCIDENTAL_NOTEHEAD_THRESHOLD_X or dist_y > ACCIDENTAL_NOTEHEAD_THRESHOLD_Y:
            return False
        
        # Accidentals are usually to the left of notehead
        if accidental.center_x > notehead.center_x + ACCIDENTAL_NOTEHEAD_THRESHOLD_X:
            return False
        
        return True
    
    def _refine_with_mlp(self, notehead: Symbol, geometric_links: List[Symbol], 
                        all_candidates: List[Symbol]) -> List[Symbol]:
        """
        Stage 2: MLP refinement.
        Validates geometric links and adds high-confidence MLP links.
        Ensures each notehead connects to at most one stem.
        
        Args:
            notehead: The notehead symbol
            geometric_links: Symbols linked by geometric rules
            all_candidates: All candidate symbols to check
            
        Returns:
            Refined list of linked symbols
        """
        if not self.linker:
            return geometric_links
        
        refined = []
        geometric_ids = {id(link) for link in geometric_links}
        
        # Separate stems from other links
        geometric_stems = [link for link in geometric_links if 'stem' in link.class_name.lower()]
        geometric_others = [link for link in geometric_links if 'stem' not in link.class_name.lower()]
        
        # Validate geometric stem with MLP (keep only the best one)
        validated_stem = None
        best_stem_prob = 0.0
        
        for stem in geometric_stems:
            prob = self.linker.predict(notehead, stem, self.image_shape)
            if prob > 0.3 and prob > best_stem_prob:  # MLP confirms and it's the best
                best_stem_prob = prob
                validated_stem = stem
        
        # If no geometric stem was validated, look for MLP-only stems
        if validated_stem is None:
            window_x = 150
            window_y = 400
            
            for cand in all_candidates:
                if cand is notehead or id(cand) in geometric_ids:
                    continue
                
                if 'stem' not in cand.class_name.lower():
                    continue
                
                if abs(notehead.center_x - cand.center_x) > window_x:
                    continue
                if abs(notehead.center_y - cand.center_y) > window_y:
                    continue
                
                prob = self.linker.predict(notehead, cand, self.image_shape)
                if prob > 0.5 and prob > best_stem_prob:  # High confidence and better than previous
                    best_stem_prob = prob
                    validated_stem = cand
        
        # Add validated stem if found
        if validated_stem is not None:
            refined.append(validated_stem)
            geometric_ids.add(id(validated_stem))
        
        # Validate other geometric links (non-stems)
        # IMPORTANT: Flags should only link through stems, not directly to noteheads
        for link in geometric_others:
            # Skip flags that might have been incorrectly linked directly to notehead
            # Flags should only be linked through stems
            if 'flag' in link.class_name.lower():
                # Check if there's a stem in the links - if yes, flag is valid (linked through stem)
                # If no stem, skip this flag (shouldn't be directly linked to notehead)
                has_stem = any('stem' in l.class_name.lower() for l in geometric_links)
                if not has_stem:
                    continue  # Skip flag if no stem present
            
            prob = self.linker.predict(notehead, link, self.image_shape)
            if prob > 0.3:  # MLP confirms geometric link
                refined.append(link)
        
        # Add high-confidence MLP links that weren't found geometrically (non-stems only)
        # IMPORTANT: Flags should only link through stems, not directly to noteheads
        window_x = 150
        window_y = 400
        
        for cand in all_candidates:
            if cand is notehead or id(cand) in geometric_ids:
                continue
            
            # Skip stems (already handled above)
            if 'stem' in cand.class_name.lower():
                continue
            
            # IMPORTANT: Flags should NOT link directly to noteheads
            # Flags should only link through stems (already handled in geometric stage)
            if 'flag' in cand.class_name.lower():
                continue
            
            # Skip unrelated classes
            relevant_types = ['beam', 'dot', 'accidental', 'sharp', 'flat', 'natural']
            if not any(t in cand.class_name.lower() for t in relevant_types):
                continue
            
            if abs(notehead.center_x - cand.center_x) > window_x:
                continue
            if abs(notehead.center_y - cand.center_y) > window_y:
                continue
            
            prob = self.linker.predict(notehead, cand, self.image_shape)
            if prob > 0.5:  # High confidence MLP link
                refined.append(cand)
        
        return refined

    def _calculate_duration(self, notehead: Symbol, linked_syms: List[Symbol]) -> float:
        """
        Determines note duration based on notehead type and linked symbols.
        Uses comprehensive lookup table approach with support for:
        - Empty vs filled noteheads
        - Stem presence
        - Flag types (8th, 16th)
        - Beam layers (single, double)
        - Dots (augmentation)
        - Grace notes
        
        Args:
            notehead: The notehead symbol
            linked_syms: List of symbols linked to the notehead
            
        Returns:
            Duration in quarter note units (e.g., 1.0 = quarter, 0.5 = eighth)
        """
        # Fallback: if no links found, use geometric proximity
        if not linked_syms:
            linked_syms = self._find_nearby_symbols_geometric(notehead)
        
        # Check for grace note
        is_grace = 'grace' in notehead.class_name.lower()
        if is_grace:
            # Grace notes are typically very short (e.g., 0.125 = 32nd note)
            return 0.125
        
        # 1. Determine notehead type
        name_lower = notehead.class_name.lower()
        is_empty = ('empty' in name_lower or 
                   'half' in name_lower or 
                   'whole' in name_lower or
                   'notehead-empty' in name_lower)
        
        # 2. Check for stem
        has_stem = any('stem' in s.class_name.lower() for s in linked_syms)
        
        # 3. Base duration lookup table
        duration = 1.0  # Default: quarter note
        
        if is_empty:
            # Empty notehead (whole or half note)
            if has_stem:
                duration = 2.0  # Half note (empty head + stem)
            else:
                duration = 4.0  # Whole note (empty head, no stem)
        else:
            # Filled notehead (quarter, eighth, sixteenth, etc.)
            # Count flags by type
            num_flags_8 = 0
            num_flags_16 = 0
            num_flags_32 = 0
            
            for sym in linked_syms:
                sym_name = sym.class_name.lower()
                if '16th' in sym_name or 'flag16' in sym_name:
                    num_flags_16 += 1
                elif '32nd' in sym_name or 'flag32' in sym_name:
                    num_flags_32 += 1
                elif '8th' in sym_name or ('flag' in sym_name and '16' not in sym_name and '32' not in sym_name):
                    num_flags_8 += 1
            
            # Count beam layers
            beam_symbols = [s for s in linked_syms if 'beam' in s.class_name.lower()]
            num_beam_layers = len(beam_symbols)
            
            # Duration determination priority:
            # 1. Flags (most specific)
            # 2. Beams (less specific, but common)
            # 3. Default (quarter note)
            
            if num_flags_32 > 0:
                duration = 0.125  # 32nd note
            elif num_flags_16 > 0:
                duration = 0.25   # 16th note
            elif num_flags_8 > 0:
                duration = 0.5    # 8th note
            elif num_beam_layers > 0:
                # Beamed notes: typically 8th notes, but can be 16th with 2 layers
                if num_beam_layers >= 2:
                    duration = 0.25  # Double beam = 16th note
                else:
                    duration = 0.5   # Single beam = 8th note
            else:
                duration = 1.0  # Quarter note (filled head + stem, no flags/beams)
        
        # 4. Check for augmentation dot
        has_dot = any('dot' in s.class_name.lower() for s in linked_syms)
        if has_dot:
            duration *= 1.5  # Dotted note: original duration * 1.5
        
        return duration
    
    def _find_nearby_symbols_geometric(self, notehead: Symbol) -> List[Symbol]:
        """
        Fallback method: find nearby symbols using geometric rules.
        Delegates to _find_linked_symbols_geometric for consistency.
        """
        # Use the same geometric logic, but search in all symbols
        return self._find_linked_symbols_geometric(notehead, self.symbols)

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
    
    def _is_symbol_in_region(self, symbol: Symbol, x_range: Tuple[float, float], 
                            y_range: Tuple[float, float]) -> bool:
        """
        Checks if a symbol is within the specified region.
        
        Args:
            symbol: The symbol to check
            x_range: (x_min, x_max) tuple
            y_range: (y_min, y_max) tuple
            
        Returns:
            True if symbol center is within the region
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        return (x_min <= symbol.center_x <= x_max and 
                y_min <= symbol.center_y <= y_max)
    
    def _calculate_vertical_distance(self, sym1: Symbol, sym2: Symbol) -> float:
        """
        Calculates vertical distance between two symbols.
        
        Args:
            sym1: First symbol
            sym2: Second symbol
            
        Returns:
            Absolute vertical distance in pixels
        """
        return abs(sym1.center_y - sym2.center_y)
    
    def _is_vertically_aligned(self, sym1: Symbol, sym2: Symbol, threshold: float = 10.0) -> bool:
        """
        Checks if two symbols are vertically aligned (similar x-coordinate).
        
        Args:
            sym1: First symbol
            sym2: Second symbol
            threshold: Maximum horizontal distance to consider aligned
            
        Returns:
            True if symbols are vertically aligned
        """
        return abs(sym1.center_x - sym2.center_x) < threshold
    
    def _is_horizontally_overlapping(self, sym1: Symbol, sym2: Symbol) -> bool:
        """
        Checks if two symbols overlap horizontally (x-axis).
        
        Args:
            sym1: First symbol
            sym2: Second symbol
            
        Returns:
            True if symbols overlap horizontally
        """
        return not (sym1.x2 < sym2.x1 or sym2.x2 < sym1.x1)
    
    def _get_notes_in_x_range(self, notes: List[AssembledNote], x_start: float, 
                              x_end: float, tolerance: float = 10.0) -> List[AssembledNote]:
        """
        Gets notes whose notehead center X coordinate falls within the range.
        Uses a small tolerance to account for detection errors.
        
        Args:
            notes: List of notes to filter
            x_start: Start X coordinate
            x_end: End X coordinate
            tolerance: Tolerance in pixels for boundary matching (default: 10.0)
            
        Returns:
            List of notes in the X range
        """
        result = []
        for note in notes:
            if note.original_symbol:
                note_x = note.original_symbol.center_x
                # Use tolerance: note can be slightly outside bracket range due to detection errors
                if (x_start - tolerance) <= note_x <= (x_end + tolerance):
                    result.append(note)
        return result
    
    def _get_staff_height(self, system: StaffSystem) -> float:
        """
        Gets the height of a staff system.
        
        Args:
            system: The staff system
            
        Returns:
            Height in pixels (distance between top and bottom lines)
        """
        if system.lines and len(system.lines) >= 2:
            return system.lines[-1] - system.lines[0]
        return system.avg_spacing * 4  # Fallback: 4 line spacings
    
    def _find_nearby_numeral_3(self, symbol: Symbol, all_symbols: List[Symbol], 
                               search_radius: float) -> List[Symbol]:
        """
        Finds numeral_3 symbols near a given symbol.
        
        Args:
            symbol: Reference symbol
            all_symbols: All symbols to search
            search_radius: Search radius in pixels
            
        Returns:
            List of numeral_3 symbols within search radius
        """
        result = []
        for sym in all_symbols:
            sym_name_lower = sym.class_name.lower()
            if 'numeral_3' in sym_name_lower or ('3' in sym_name_lower and 'numeral' in sym_name_lower):
                # Calculate distance
                dx = sym.center_x - symbol.center_x
                dy = sym.center_y - symbol.center_y
                distance = np.sqrt(dx * dx + dy * dy)
                if distance <= search_radius:
                    result.append(sym)
        return result
    
    def _detect_triplets_rule1_bracket(self, measure: AssembledMeasure, all_symbols: List[Symbol], 
                                      system: StaffSystem, all_processed_notes: List[AssembledNote] = None) -> List[List[AssembledNote]]:
        """
        Rule 1: Bracket-based triplet detection (highest priority).
        Detects triplets based on tuple_bracket/line symbols.
        Searches across all processed measures (not just current measure) as per rule table.
        
        Args:
            measure: The measure to check
            all_symbols: All symbols in the system
            system: The staff system
            all_processed_notes: All notes from previously processed measures (for cross-measure search)
            
        Returns:
            List of triplet groups, each group is a list of 3 AssembledNote objects
        """
        
        triplet_groups = []
        staff_height = self._get_staff_height(system)
        
        # Find all tuple_bracket/line symbols
        bracket_symbols = [s for s in all_symbols 
                          if 'tuple_bracket' in s.class_name.lower() or 
                          'tuple_line' in s.class_name.lower()]
        
        
        for bracket in bracket_symbols:
            # Get bracket X-axis range
            x_start = bracket.x1
            x_end = bracket.x2
            
            # Find notes whose notehead center X falls within bracket range
            # Search only within current measure (not across measures)
            
            candidate_notes = self._get_notes_in_x_range(measure.notes, x_start, x_end)
            candidate_count_before_dedup = len(candidate_notes)
            
            # Remove duplicates based on note ID and X coordinate
            # Same note object should not appear twice, and notes with same X should be deduplicated
            seen_note_ids = set()
            seen_x_coords = set()
            unique_candidate_notes = []
            for note in candidate_notes:
                note_id = id(note)
                note_x = float(note.x) if note.original_symbol else None
                
                # Check both note ID and X coordinate to avoid duplicates
                if note_id not in seen_note_ids:
                    if note_x is not None:
                        # Round X coordinate to avoid floating point precision issues
                        note_x_rounded = round(note_x, 1)
                        if note_x_rounded not in seen_x_coords:
                            seen_note_ids.add(note_id)
                            seen_x_coords.add(note_x_rounded)
                            unique_candidate_notes.append(note)
                    else:
                        # Note without original_symbol, just check ID
                        seen_note_ids.add(note_id)
                        unique_candidate_notes.append(note)
            candidate_notes = unique_candidate_notes
            candidate_count_after_dedup = len(candidate_notes)
            
            
            # Filter: remove notes that are too far vertically (more than one staff height)
            filtered_notes = []
            for note in candidate_notes:
                if note.original_symbol:
                    vertical_dist = self._calculate_vertical_distance(bracket, note.original_symbol)
                    if vertical_dist <= staff_height * 1.5:
                        filtered_notes.append(note)
                else:
                    pass
            
            
            # Search for numeral_3 near bracket center (required for triplet detection)
            bracket_center_x = bracket.center_x
            bracket_center_y = bracket.center_y
            search_radius = staff_height * 0.5  # Search within half staff height
            
            found_numeral_3 = False
            for sym in all_symbols:
                sym_name_lower = sym.class_name.lower()
                if 'numeral_3' in sym_name_lower or ('3' in sym_name_lower and 'numeral' in sym_name_lower):
                    dx = sym.center_x - bracket_center_x
                    dy = sym.center_y - bracket_center_y
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance <= search_radius:
                        found_numeral_3 = True
                        break
            
            # Only mark as triplet if we have exactly 3 notes.
            # Relaxed Rule: numeral_3 is OPTIONAL if we have exactly 3 notes inside a tuple bracket.
            if len(filtered_notes) == 3:
                # Sort notes by X coordinate
                filtered_notes.sort(key=lambda n: n.x)
                triplet_groups.append(filtered_notes)
        
        return triplet_groups
    
    def _detect_triplets_rule2_beam(self, measure: AssembledMeasure, all_symbols: List[Symbol], 
                                    system: StaffSystem, processed_notes: set) -> List[List[AssembledNote]]:
        """
        Rule 2: Beam-based triplet detection (second priority).
        Detects triplets based on beams linking 3 or multiples of 3 stems.
        Skips if time signature is 6/8, 9/8, or 12/8 (compound time signatures).
        
        Args:
            measure: The measure to check
            all_symbols: All symbols in the system
            system: The staff system
            processed_notes: Set of note IDs that have already been processed by higher priority rules
            
        Returns:
            List of triplet groups, each group is a list of AssembledNote objects
        """
        # Skip Rule 2 if time signature is 6/8, 9/8, or 12/8 (compound time signatures)
        if measure.time_signature:
            time_sig = measure.time_signature.strip()
            if time_sig in ['6/8', '9/8', '12/8']:
                return []
        
        triplet_groups = []
        
        # Find all beams and count how many stems they link to
        beam_to_stems = {}  # beam_id -> list of stem_ids
        beam_id_to_beam = {}  # beam_id -> beam Symbol
        stem_id_to_note = {}  # stem_id -> AssembledNote
        
        for note in measure.notes:
            if not note.original_symbol or id(note) in processed_notes:
                continue
            
            # Include rests in triplet detection (treat them as notes)
            # For rests, we don't have stems, so skip beam-based detection for rests
            # (rests are handled in other rules)
            is_rest = note.pitch is None or 'rest' in note.original_symbol.class_name.lower()
            if is_rest:
                # Rests don't have stems, so they can't be part of beam-based triplets
                continue
            
            # Find beams linked to this note's stem
            linked_stem = None
            linked_stem_id = None
            for linked_sym in note.linked_symbols:
                if 'stem' in linked_sym.class_name.lower():
                    linked_stem = linked_sym
                    linked_stem_id = id(linked_stem)
                    stem_id_to_note[linked_stem_id] = note
                    break
            
            if not linked_stem:
                continue
            
            # Find beams linked to this stem
            for linked_sym in note.linked_symbols:
                if 'beam' in linked_sym.class_name.lower():
                    beam_id = id(linked_sym)
                    beam_id_to_beam[beam_id] = linked_sym
                    
                    if beam_id not in beam_to_stems:
                        beam_to_stems[beam_id] = []
                    if linked_stem_id not in beam_to_stems[beam_id]:
                        beam_to_stems[beam_id].append(linked_stem_id)
        
        # Check each beam that links to 3 or multiples of 3 stems
        for beam_id, stem_ids in beam_to_stems.items():
            num_stems = len(stem_ids)
            if num_stems >= 3 and num_stems % 3 == 0:
                beam = beam_id_to_beam[beam_id]
                
                # Get notes corresponding to these stems
                notes = []
                for stem_id in stem_ids:
                    if stem_id in stem_id_to_note:
                        note = stem_id_to_note[stem_id]
                        if id(note) not in processed_notes:
                            notes.append(note)
                
                # For triplets, we need exactly 3 notes
                if len(notes) >= 3:
                    # Take first 3 notes (sorted by X coordinate)
                    notes.sort(key=lambda n: n.x)
                    triplet_candidates = notes[:3]
                    
                    # Define search region: center of beam, radius = 3 * line spacing
                    beam_center_x = (beam.x1 + beam.x2) / 2
                    beam_center_y = (beam.y1 + beam.y2) / 2
                    search_radius = system.avg_spacing * 3
                    
                    # Find numeral_3 in search region
                    found_numeral_3 = None
                    for sym in all_symbols:
                        sym_name_lower = sym.class_name.lower()
                        if 'numeral_3' in sym_name_lower or ('3' in sym_name_lower and 'numeral' in sym_name_lower):
                            dx = sym.center_x - beam_center_x
                            dy = sym.center_y - beam_center_y
                            distance = np.sqrt(dx * dx + dy * dy)
                            if distance <= search_radius:
                                # Collision check: numeral_3 should overlap with beam or be above/below it
                                overlaps_beam = self._check_bbox_overlap(sym, beam) or \
                                               (sym.y2 < beam.y1 and abs(sym.center_x - beam_center_x) < beam.width) or \
                                               (sym.y1 > beam.y2 and abs(sym.center_x - beam_center_x) < beam.width)
                                
                                if overlaps_beam:
                                    found_numeral_3 = sym
                                    break
                    
                    if found_numeral_3:
                        # Finger number exclusion: check if numeral_3 is strictly vertically aligned
                        # with a single notehead and very close (likely finger number)
                        is_finger_number = False
                        for note in triplet_candidates:
                            if note.original_symbol:
                                # Check if numeral_3 is directly above notehead
                                if self._is_vertically_aligned(found_numeral_3, note.original_symbol, threshold=5.0):
                                    vertical_dist = self._calculate_vertical_distance(found_numeral_3, note.original_symbol)
                                    if vertical_dist < system.avg_spacing * 0.5:  # Very close
                                        is_finger_number = True
                                        break
                        
                        if not is_finger_number:
                            # Check if numeral_3 is at geometric center X position of the note group
                            note_center_x = np.mean([n.x for n in triplet_candidates])
                            if abs(found_numeral_3.center_x - note_center_x) < system.avg_spacing * 1.5:
                                triplet_groups.append(triplet_candidates)
        
        return triplet_groups
    
    def _detect_triplets_rule3_loose(self, measure: AssembledMeasure, all_symbols: List[Symbol], 
                                     system: StaffSystem, processed_notes: set, 
                                     used_numeral_3: set) -> List[List[AssembledNote]]:
        """
        Rule 3: Numeral-3-centered triplet detection (improved).
        Detects triplets by finding 3 notes closest to each numeral_3 symbol using Euclidean distance.
        This is a more direct and robust approach that centers on the numeral_3 symbol.
        
        Args:
            measure: The measure to check
            all_symbols: All symbols in the system
            system: The staff system
            processed_notes: Set of note IDs already processed by higher priority rules
            used_numeral_3: Set of numeral_3 symbol IDs already used by other rules
            
        Returns:
            List of triplet groups, each group is a list of 3 AssembledNote objects
        """
        triplet_groups = []
        
        # Find all numeral_3 symbols, excluding those that are part of time signatures
        unused_numeral_3 = []
        for sym in all_symbols:
            sym_name_lower = sym.class_name.lower()
            if ('numeral_3' in sym_name_lower or ('3' in sym_name_lower and 'numeral' in sym_name_lower)) and \
               id(sym) not in used_numeral_3:
                # Check if this numeral_3 is part of a time signature
                # Time signatures are vertical stacks of numerals (e.g., 3/4, 3/8, 2/3, 4/3)
                is_time_sig_numeral = False
                
                # Look for other numerals that are vertically aligned with this numeral_3
                # and form a time signature pattern (top/bottom stack)
                for other_sym in all_symbols:
                    if other_sym == sym:
                        continue
                    other_name_lower = other_sym.class_name.lower()
                    if 'numeral' in other_name_lower:
                        # Check if vertically aligned (within threshold)
                        x_threshold = system.avg_spacing * 0.8  # Tight alignment for time signatures
                        if abs(sym.center_x - other_sym.center_x) < x_threshold:
                            # Check if they form a vertical stack
                            # Time signature: top number above bottom number, within reasonable distance
                            vertical_dist = abs(sym.center_y - other_sym.center_y)
                            if vertical_dist < system.avg_spacing * 4 and vertical_dist > system.avg_spacing * 0.5:
                                # Extract digit from other symbol
                                other_digit = None
                                for digit in range(10):
                                    if f'numeral_{digit}' in other_name_lower or f'numeral{digit}' in other_name_lower:
                                        other_digit = digit
                                        break
                                
                                if other_digit is not None:
                                    # Check if this forms a time signature pattern
                                    # Common time signatures: 3/4, 3/8, 2/3, 4/3, etc.
                                    # Either numeral_3 is on top (3/X) or on bottom (X/3)
                                    if sym.center_y < other_sym.center_y:
                                        # numeral_3 is on top (3/X format)
                                        # Common denominators: 2, 4, 8, 16
                                        if other_digit in [2, 4, 8, 16]:
                                            is_time_sig_numeral = True
                                            break
                                    elif sym.center_y > other_sym.center_y:
                                        # numeral_3 is on bottom (X/3 format)
                                        # Common numerators: 2, 4, 8, 16
                                        if other_digit in [2, 4, 8, 16]:
                                            is_time_sig_numeral = True
                                            break
                
                # Only add if not part of time signature
                if not is_time_sig_numeral:
                    unused_numeral_3.append(sym)
        
        # Get all notes in the measure (including processed ones for search, but we'll filter later)
        # We need to search all notes to find complete triplets, even if some were processed by other rules
        all_measure_notes = [n for n in measure.notes if n.original_symbol]
        
        # For each numeral_3, find the 3 closest notes using Euclidean distance
        for numeral_3 in unused_numeral_3:
            numeral_3_x = numeral_3.center_x
            numeral_3_y = numeral_3.center_y
            
            # Calculate Euclidean distance from numeral_3 to each note
            notes_with_distance = []
            # Use a generous search radius: 6 * avg_spacing (covers about 1.5 staff heights)
            max_distance = system.avg_spacing * 6
            
            for note in all_measure_notes:
                if note.original_symbol:
                    note_x = note.original_symbol.center_x
                    note_y = note.original_symbol.center_y
                    
                    # Calculate Euclidean distance
                    dx = note_x - numeral_3_x
                    dy = note_y - numeral_3_y
                    distance = np.sqrt(dx * dx + dy * dy)
                    
                    if distance <= max_distance:
                        notes_with_distance.append((note, distance))
            
            # Check for finger number (Rule 3 enhancement)
            # If numeral_3 is strictly vertically aligned with a notehead and very close, it's likely a finger number
            is_finger_number = False
            for note in all_measure_notes:
                if note.original_symbol:
                    # Check if numeral_3 is directly above/below notehead
                    if self._is_vertically_aligned(numeral_3, note.original_symbol, threshold=5.0):
                        vertical_dist = self._calculate_vertical_distance(numeral_3, note.original_symbol)
                        # Identify as finger number if vertically aligned and close (less than 1 staff height)
                        if vertical_dist < system.avg_spacing * 4: 
                             is_finger_number = True
                             break
            
            if is_finger_number:
                continue

            # If we have at least 3 notes, select the 3 closest ones
            if len(notes_with_distance) >= 3:
                # Sort by distance and take the 3 closest notes
                notes_with_distance.sort(key=lambda x: x[1])
                closest_3 = [note for note, _ in notes_with_distance[:3]]
                
                # Filter: only use notes that haven't been processed yet
                # This ensures we don't create incomplete triplets
                unprocessed_closest_3 = [n for n in closest_3 if id(n) not in processed_notes]
                
                # Only add if we have exactly 3 unprocessed notes
                # This ensures complete triplets
                if len(unprocessed_closest_3) == 3:
                    # Sort the 3 notes by X coordinate for consistency
                    unprocessed_closest_3.sort(key=lambda n: n.x)
                    

                    
                    # Add to triplet groups
                    triplet_groups.append(unprocessed_closest_3)
        
        return triplet_groups
    
    def _apply_triplet_modifications(self, triplet_group: List[AssembledNote], rule_triggered: str) -> None:
        """
        Rule 4: Apply triplet modifications to a group of notes.
        Modifies duration and sets tuplet attributes for MusicXML generation.
        
        Args:
            triplet_group: List of 3 notes that form a triplet
            rule_triggered: Which rule triggered this triplet ("Rule1", "Rule2", "Rule3", or "Rule5")
        """
        
        notes_modified = 0
        for note in triplet_group:
            # Save base duration
            note.base_duration = note.duration
            
            # Modify duration: base_duration * 2/3
            note.duration = note.base_duration * (2.0 / 3.0)
            
            # Set time modification parameters
            note.time_modification_actual_notes = 3
            note.time_modification_normal_notes = 2
            
            # Set tuplet flags
            note.is_tuplet = True
            note.tuplet_type = "triplet"
            note.tuplet_rule_triggered = rule_triggered
            
            notes_modified += 1
            
            # Set bracket display logic
            if rule_triggered == "Rule1":
                note.tuplet_bracket = True  # Bracket-based: show bracket
                note.tuplet_confidence = "High"
            elif rule_triggered == "Rule2":
                note.tuplet_bracket = False  # Beam-based: no bracket needed
                note.tuplet_confidence = "High"
            elif rule_triggered == "Rule3":
                note.tuplet_bracket = True  # Loose number: show bracket for clarity
                note.tuplet_confidence = "High"
            elif rule_triggered == "Rule5":
                note.tuplet_bracket = True  # Sanity check: show bracket
                note.tuplet_confidence = "Low"
        

    
    def _parse_time_signature_duration(self, time_signature: Optional[str]) -> float:
        """
        Parses time signature string to get expected duration in quarter note units.
        
        Args:
            time_signature: Time signature string (e.g., "4/4", "3/4", "C")
            
        Returns:
            Expected duration in quarter note units, or 4.0 as default
        """
        if not time_signature:
            return 4.0  # Default: 4/4
        
        if '/' in time_signature:
            try:
                numerator, denominator = time_signature.split('/')
                numerator = float(numerator)
                denominator = float(denominator)
                # Convert to quarter note units
                # e.g., 4/4 = 4.0, 3/4 = 3.0, 6/8 = 3.0 (assuming 6/8 = 3 quarter notes)
                return numerator * (4.0 / denominator)
            except:
                return 4.0
        elif time_signature == "C":
            return 4.0  # Common time = 4/4
        else:
            return 4.0  # Default
    
    def _detect_triplets_rule5_sanity_check(self, measure: AssembledMeasure, all_symbols: List[Symbol], 
                                            system: StaffSystem, processed_notes: set) -> List[List[AssembledNote]]:
        """
        Rule 5: Sanity check (fallback mechanism).
        Detects implicit triplets by checking if measure duration exceeds time signature.
        
        Args:
            measure: The measure to check
            all_symbols: All symbols in the system
            system: The staff system
            processed_notes: Set of note IDs already processed by other rules
            
        Returns:
            List of triplet groups detected by sanity check
        """
        triplet_groups = []
        
        # Skip if no time signature
        if not measure.time_signature:
            return triplet_groups
        
        # Calculate current measure duration
        current_duration = sum(note.duration for note in measure.notes)
        
        # Get expected duration from time signature
        expected_duration = self._parse_time_signature_duration(measure.time_signature)
        
        # Check if duration exceeds expected (with small tolerance)
        tolerance = 0.1
        if current_duration <= expected_duration + tolerance:
            return triplet_groups  # No overflow, no need for correction
        
        # Find unprocessed notes sorted by X coordinate
        unprocessed_notes = [n for n in measure.notes 
                            if n.original_symbol and id(n) not in processed_notes]
        unprocessed_notes.sort(key=lambda n: n.x)
        
        # Look for 3 consecutive notes with same base duration
        for i in range(len(unprocessed_notes) - 2):
            note_group = unprocessed_notes[i:i+3]
            
            # Check if all have same base duration (or current duration if base not set)
            base_durations = [n.base_duration if n.base_duration is not None else n.duration 
                            for n in note_group]
            if len(set(base_durations)) > 1:
                continue  # Durations must be identical
            
            base_duration = base_durations[0]
            
            # Check if converting these 3 notes to triplets would fix the overflow
            # Original total: 3 * base_duration
            # Triplet total: 3 * (base_duration * 2/3) = 2 * base_duration
            # Reduction: base_duration
            reduction = base_duration
            
            # Check if this reduction would bring duration within expected range
            new_duration = current_duration - reduction
            if abs(new_duration - expected_duration) < tolerance:
                # Check if there's no numeral_3 detected in this region
                # (if there was, it should have been caught by other rules)
                note_x_start = min(n.x for n in note_group)
                note_x_end = max(n.x for n in note_group)
                note_y_center = np.mean([n.original_symbol.center_y for n in note_group if n.original_symbol])
                
                has_numeral_3 = False
                for sym in all_symbols:
                    sym_name_lower = sym.class_name.lower()
                    if 'numeral_3' in sym_name_lower or ('3' in sym_name_lower and 'numeral' in sym_name_lower):
                        if note_x_start <= sym.center_x <= note_x_end:
                            vertical_dist = abs(sym.center_y - note_y_center)
                            if vertical_dist < system.avg_spacing * 3:
                                has_numeral_3 = True
                                break
                
                # If no numeral_3 found, this might be an implicit triplet
                if not has_numeral_3:
                    triplet_groups.append(note_group)
        
        return triplet_groups
    
    def _detect_triplets(self, measure: AssembledMeasure, all_symbols: List[Symbol], system: StaffSystem, all_processed_notes: List[AssembledNote] = None) -> None:
        """
        Main triplet detection dispatcher.
        Applies all triplet detection rules in priority order (Rule1 > Rule2 > Rule3 > Rule5).
        
        Args:
            measure: The measure to check for triplets
            all_symbols: All symbols in the system
            system: The staff system
            all_processed_notes: All notes from previously processed measures (for Rule 1 cross-measure search)
        """
        if all_processed_notes is None:
            all_processed_notes = measure.notes
        # Track processed notes and used symbols to avoid duplicate processing
        processed_notes = set()  # Set of note IDs
        used_numeral_3 = set()  # Set of numeral_3 symbol IDs
        
        # Rule 1: Bracket-based detection (highest priority)
        triplet_groups_rule1 = self._detect_triplets_rule1_bracket(measure, all_symbols, system, all_processed_notes)
        for group_idx, group in enumerate(triplet_groups_rule1):
            # Filter: only process notes that haven't been processed yet
            # This prevents Rule1 from overwriting Rule3's markings
            unprocessed_group = [n for n in group if id(n) not in processed_notes]
            
            # Only apply if we have exactly 3 unprocessed notes
            # If some notes are already processed, skip this group to avoid incomplete triplets
            if len(unprocessed_group) == 3:
                # Mark notes as processed
                for note in unprocessed_group:
                    processed_notes.add(id(note))
                # Apply modifications
                self._apply_triplet_modifications(unprocessed_group, "Rule1")
            else:
                pass

            # Mark any numeral_3 symbols used (if we can find them)
            # Note: Rule1 searches for numeral_3 but doesn't track which one, so we skip tracking here
        
        # Rule 2: Beam-based detection (third priority)
        triplet_groups_rule2 = self._detect_triplets_rule2_beam(measure, all_symbols, system, processed_notes)
        for group in triplet_groups_rule2:
            # Mark notes as processed
            for note in group:
                processed_notes.add(id(note))
            # Apply modifications
            self._apply_triplet_modifications(group, "Rule2")
            # Note: Rule2 finds numeral_3 but doesn't return it, so we skip tracking here
        
        # Rule 3: Numeral-3-centered detection (now lowest priority)
        # This rule now runs last to avoid aggressive false positives (e.g., finger numbers)
        # It only picks up what's left after Bracket and Beam rules
        triplet_groups_rule3 = self._detect_triplets_rule3_loose(measure, all_symbols, system, processed_notes, used_numeral_3)
        for group in triplet_groups_rule3:
            # Mark notes as processed
            for note in group:
                processed_notes.add(id(note))
            # Apply modifications
            self._apply_triplet_modifications(group, "Rule3")
        
        # Rule 5: Sanity check (fallback, called after all other rules)
        # This is called after all notes are processed, so we check the entire measure
        triplet_groups_rule5 = self._detect_triplets_rule5_sanity_check(measure, all_symbols, system, processed_notes)
        for group in triplet_groups_rule5:
            # Mark notes as processed
            for note in group:
                processed_notes.add(id(note))
            # Apply modifications
            self._apply_triplet_modifications(group, "Rule5")
    
    def _detect_clef(self, measure_symbols: List[Symbol], system: StaffSystem, 
                     current_clef: ClefType, is_first_measure: bool) -> Optional[ClefType]:
        """
        Implements Rule 4 from rule.md: Clef Detection with state persistence.
        
        Args:
            measure_symbols: Symbols in the current measure
            system: The staff system
            current_clef: Current clef state (to inherit if not found)
            is_first_measure: True if this is the first measure
            
        Returns:
            Detected clef, or None if should inherit
        """
        # Search for clef symbols in the measure
        for sym in measure_symbols:
            class_name_lower = sym.class_name.lower()
            if any(keyword in class_name_lower for keyword in ['g-clef', 'f-clef', 'c-clef', 'clef']):
                # Check if symbol is within the staff system's vertical range
                if system.contains_y(sym.center_y, margin=system.avg_spacing):
                    detected_clef = PitchEngine.get_clef_from_name(sym.class_name)
                    return detected_clef
        
        # If not found and it's the first measure, use default
        if is_first_measure:
            # Default: Part 1 = Treble, Part 2 = Bass (can be refined later)
            return ClefType.G_CLEF  # Default, but mark as low confidence
        
        # Otherwise, inherit (return None to keep current state)
        return None
    
    def _detect_key_signature(self, measure_symbols: List[Symbol], system: StaffSystem,
                              current_key: Optional[str]) -> Optional[str]:
        """
        Implements Rule 5 from rule.md: Key Signature Detection.
        
        Args:
            measure_symbols: Symbols in the current measure
            system: The staff system
            current_key: Current key signature state (to inherit if not found)
            
        Returns:
            Detected key signature string, or None if should inherit
        """
        # Mode A: Check for key_signature label
        for sym in measure_symbols:
            class_name_lower = sym.class_name.lower()
            if 'key_signature' in class_name_lower:
                # Try to extract key from class name or use default
                # For now, return a placeholder - can be enhanced with ML-based parsing
                return "C"  # Default to C Major/A Minor
        
        # Mode B: Cluster discrete sharp/flat/natural symbols
        # Find clef position (usually at the start of measure)
        clef_x = None
        for sym in measure_symbols:
            if 'clef' in sym.class_name.lower():
                clef_x = sym.center_x
                break
        
        if clef_x is None:
            return None  # Can't determine region without clef
        
        # Find first note position (to define search region)
        first_note_x = None
        for sym in measure_symbols:
            if 'note' in sym.class_name.lower() or 'head' in sym.class_name.lower():
                first_note_x = sym.center_x
                break
        
        # Collect accidentals in the key signature region (between clef and first note)
        accidentals = []
        avg_note_width = system.avg_spacing * 2  # Approximate note width
        
        # Define search region: if first_note_x exists, use it; otherwise use a wider margin
        if first_note_x is not None:
            search_end_x = first_note_x
        else:
            # If no note found, search up to 10 note widths after clef
            search_end_x = clef_x + 10 * avg_note_width
        
        for sym in measure_symbols:
            class_name_lower = sym.class_name.lower()
            if any(keyword in class_name_lower for keyword in ['sharp', 'flat', 'natural']):
                # Check if in the key signature region
                # If first_note_x exists, use strict < to avoid including accidentals at note position
                # If first_note_x doesn't exist, use <= to be more inclusive
                if first_note_x is not None:
                    in_region = clef_x < sym.center_x < first_note_x
                else:
                    in_region = clef_x < sym.center_x <= search_end_x
                
                if in_region:
                    # Check if within staff system
                    if system.contains_y(sym.center_y, margin=system.avg_spacing * 2):
                        accidentals.append(sym)
        
        if not accidentals:
            return None  # No accidentals found, inherit
        
        # Cluster accidentals by X position (within 1.5 note width)
        accidentals.sort(key=lambda s: s.center_x)
        clusters = []
        current_cluster = [accidentals[0]]
        
        for i in range(1, len(accidentals)):
            if accidentals[i].center_x - current_cluster[-1].center_x < 1.5 * avg_note_width:
                current_cluster.append(accidentals[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [accidentals[i]]
        clusters.append(current_cluster)
        
        # Count sharps and flats
        num_sharps = 0
        num_flats = 0
        
        for cluster in clusters:
            # Check if cluster contains sharps or flats
            has_sharp = any('sharp' in s.class_name.lower() for s in cluster)
            has_flat = any('flat' in s.class_name.lower() for s in cluster)
            
            if has_sharp:
                num_sharps += len([s for s in cluster if 'sharp' in s.class_name.lower()])
            elif has_flat:
                num_flats += len([s for s in cluster if 'flat' in s.class_name.lower()])
        
        # Semantic parsing (Rule 5)
        if num_sharps > 0:
            # Map number of sharps to key
            key_map = {1: "1#", 2: "2#", 3: "3#", 4: "4#", 5: "5#", 6: "6#", 7: "7#"}
            return key_map.get(num_sharps, f"{num_sharps}#")
        elif num_flats > 0:
            # Map number of flats to key
            key_map = {1: "1b", 2: "2b", 3: "3b", 4: "4b", 5: "5b", 6: "6b", 7: "7b"}
            return key_map.get(num_flats, f"{num_flats}b")
        else:
            # No accidentals = C Major / A Minor
            return "C"
    
    def _detect_time_signature(self, measure_symbols: List[Symbol], system: StaffSystem,
                               current_time: Optional[str]) -> Optional[str]:
        """
        Implements Rule 6 from rule.md: Time Signature Detection.
        
        Args:
            measure_symbols: Symbols in the current measure
            system: The staff system
            current_time: Current time signature state (to inherit if not found)
            
        Returns:
            Detected time signature string, or None if should inherit
        """
        # 1. Search for letter_c (Common Time = 4/4)
        # Only recognize as time signature if C is between staff lines
        for sym in measure_symbols:
            class_name_lower = sym.class_name.lower()
            if 'letter_c' in class_name_lower or ('letter' in class_name_lower and 'c' in class_name_lower):
                # Check if the symbol is between staff lines (within the staff system range)
                if system.contains_y(sym.center_y, margin=0):
                    return "4/4"
        
        # 2. Search for time_signature label and associated numerals
        # This replaces the blind "return 4/4" for time_signature class
        for sym in measure_symbols:
            class_name_lower = sym.class_name.lower()
            if 'time_signature' in class_name_lower:
                # Found the time signature box. Now look for numerals INSIDE or CLOSE to it.
                associated_numerals = []
                for potential_num in measure_symbols:
                    if potential_num == sym: 
                        continue
                    
                    p_name = potential_num.class_name.lower()
                    if 'numeral' in p_name:
                         # Check overlap 
                         if self._check_bbox_overlap(sym, potential_num):
                             # Extract digit
                             for digit in range(10):
                                 if f'numeral_{digit}' in p_name or f'numeral{digit}' in p_name:
                                     associated_numerals.append((potential_num, digit))
                                     break
                
                if len(associated_numerals) >= 2:
                     associated_numerals.sort(key=lambda x: x[0].center_y)
                     top_num, top_digit = associated_numerals[0]
                     bottom_num, bottom_digit = associated_numerals[1]
                     
                     # Check if bottom is below top
                     if bottom_num.center_y > top_num.center_y:
                         return f"{top_digit}/{bottom_digit}"
    
        # 3. Detect vertical digit stack (e.g., numeral_3 above numeral_4 = 3/4)
        # This acts as a fallback if no time_signature box is found
        numerals = []
        for sym in measure_symbols:
            class_name_lower = sym.class_name.lower()
            if 'numeral' in class_name_lower:
                # Extract digit from class name (e.g., "numeral_3" -> 3)
                for digit in range(10):
                    if f'numeral_{digit}' in class_name_lower or f'numeral{digit}' in class_name_lower:
                        numerals.append((sym, digit))
                        break
    
        if len(numerals) >= 2:
            # Sort by Y coordinate (top to bottom)
            numerals.sort(key=lambda x: x[0].center_y)
            
            # Check if they form a vertical stack (top number / bottom number)
            top_num, top_digit = numerals[0]
            bottom_num, bottom_digit = numerals[1]
            
            # Check if they are vertically aligned (within threshold)
            x_threshold = system.avg_spacing
            if abs(top_num.center_x - bottom_num.center_x) < x_threshold:
                # Check if bottom is below top
                if bottom_num.center_y > top_num.center_y:
                    return f"{top_digit}/{bottom_digit}"
    
        # Check for time signature mismatch (Rule 6: exception detection)
        # This will be called after notes are processed, so we'll handle it separately
        
        return None  # No time signature found, inherit
    
    def _assign_global_measure_numbers(self, parts: List[AssembledPart]) -> List[AssembledPart]:
        """
        Implements Rule 7 from rule.md: Global Measure Indexing.
        Assigns continuous measure numbers across all systems.
        
        Args:
            parts: List of parts (one per system)
            
        Returns:
            Parts with updated global measure numbers
        """
        global_index = 1
        
        # Process parts in order (they should already be sorted by system position)
        for part in parts:
            for measure in part.measures:
                # Check for multi-measure rest
                is_multi_measure_rest = False
                multi_measure_count = 1
                
                # Look for multi-measure_rest symbol in measure symbols
                # (We need to check if measure has a multi-measure rest)
                # For now, we'll check if measure has very few notes and a rest symbol
                if len(measure.notes) == 0:
                    # Could be a multi-measure rest - would need to check symbols
                    # For simplicity, assume single measure for now
                    pass
                
                if is_multi_measure_rest:
                    measure.number = global_index
                    global_index += multi_measure_count
                else:
                    measure.number = global_index
                    global_index += 1
        
        return parts
    
    def _detect_anacrusis(self, parts: List[AssembledPart]) -> None:
        """
        Implements Rule 8 from rule.md: Anacrusis (Pickup Measure) Detection.
        Checks if the first measure has fewer beats than expected by the time signature.
        
        Args:
            parts: List of parts to check
        """
        if not parts:
            return
        
        # Get the first measure from the first part
        first_part = parts[0]
        if not first_part.measures:
            return
        
        first_measure = first_part.measures[0]
        
        # Calculate total duration of notes in the first measure
        total_duration = sum(note.duration for note in first_measure.notes)
        
        # Get expected duration from time signature
        expected_duration = 4.0  # Default to 4/4
        if first_measure.time_signature:
            # Parse time signature (e.g., "4/4" -> 4.0, "3/4" -> 3.0)
            try:
                if '/' in first_measure.time_signature:
                    numerator, denominator = first_measure.time_signature.split('/')
                    expected_duration = float(numerator) / float(denominator) * 4.0
                elif first_measure.time_signature == "C":
                    expected_duration = 4.0  # Common Time = 4/4
            except:
                pass
        
        # If total duration is less than expected, mark as anacrusis
        if total_duration < expected_duration * 0.8:  # 80% threshold to account for rounding
            first_measure.is_implicit = True
            print(f"[Builder] Detected anacrusis (pickup measure): duration {total_duration:.2f} < expected {expected_duration:.2f}")

