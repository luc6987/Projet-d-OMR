# Optical Music Recognition (OMR) Assembly Rules

This document describes the algorithmic rules for assembling discrete staff lines and detected symbols into a hierarchical **Score Tree** structure, suitable for MusicXML generation.

---

## Definitions and Preprocessing

### Input Sets

* **$L = \{l_1, l_2, ...\}$**: Set of extracted staff line objects. Each $l_i$ contains attributes:
  * $y_{center}$: Vertical center coordinate
  * $h_{staff}$: Total height of the staff (5 lines)
  * $lines$: List of y-coordinates for the 5 staff lines

* **$S$**: Set of all detected symbols (bounding boxes). Each symbol $s$ is defined as:
  * $s = [x_1, x_2] \times [y_1, y_2]$
  * Additional attributes: `class_name`, `confidence`, `center_x`, `center_y`, `width`, `height`

### Auxiliary Functions

* **$Overlap_Y(A, B)$**: Y-axis projection overlap ratio between two objects
* **$Contains_Y(Container, Item)$**: Returns true if $Container.y_1 \le Item.y_{center} \le Container.y_2$
* **$Distance(A, B)$**: Bounding box distance defined as:
  * If bounding boxes overlap: $d = 0$
  * Otherwise: Euclidean distance between closest points on the two rectangles
  * Formula: $d^2 = \Delta x^2 + \Delta y^2$ where $\Delta x$ and $\Delta y$ are the minimum gaps in X and Y directions

---

## Phase 1: System & Staff Grouping

**Goal**: Partition all staff lines $L$ into **Systems** (musical rows) and determine the **Part** (voice) ordering within each system.

### Rule 1: Staff Clustering (System Detection)

Determine which staff lines belong to the same time segment (same system).

#### Priority 1: Brace/Bracket-Based Clustering [Highest Priority]
* Iterate through all symbols $s \in S$ where `class_name` contains: `multi-staff_brace`, `multi-staff_bracket`, or `staff_grouping`
* For each grouping symbol $g$:
  * Find all staff lines $l_i$ where $Overlap_Y(g, l_i) > 0.5$
  * Mark these $l_i$ as belonging to the same **System Group**

#### Priority 2: Measure Separator-Based Clustering [Second Priority]
* Iterate through all symbols with `class_name` = `measure_separator` (vertical lines connecting staves)
* If a `measure_separator`:
  * Top touches $l_a$ and bottom touches $l_b$
  * Then $l_a$, $l_b$, and all lines between them belong to the same **System Group**

#### Fallback: Ungrouped Lines
* Any staff lines not grouped by Priority 1 or 2 become independent **System Groups** (single-staff systems)

### Rule 2: Part Indexing (Voice Assignment)

Assign logical Part IDs within each system.

* For each **System** $Sys_k$:
  * Sort all staff lines by Y coordinate (top to bottom): $l_{k,1}, l_{k,2}, ..., l_{k,m}$
  * Assign Part IDs sequentially:
    * $l_{k,1} \rightarrow Part\_1$
    * $l_{k,2} \rightarrow Part\_2$
    * ...
  * **Consistency Check**: If $Sys_k$ has 2 staves and $Sys_{k+1}$ has 3 staves, log a warning and assign Part 3 to the third staff

---

## Phase 2: Measure Slicing (Barline Detection)

**Goal**: Horizontally segment each system into measures based on barline positions.

### Rule 3: Global Barline Alignment

Barlines in different parts of the same system should be aligned vertically.

#### Step 1: Collect Barlines
* Within the current system $Sys_k$ region, collect all symbols where `class_name` contains:
  * `thin_barline`, `thick_barline`, `repeat`
  * `measure_separator` (only if it does NOT span multiple systems)

#### Step 2: Validate Barlines
For each candidate barline, validate:
* **Aspect Ratio**: $\frac{width}{height} < 0.3$ (barlines are narrow)
* **Minimum Height**: $height \ge 20$ pixels
* **Staff Overlap**: Barline must overlap at least 50% of staff system height

#### Step 3: X-Axis Projection Fusion
* Due to detection errors, barlines at the same temporal position may have slightly different X coordinates
* **Merge Strategy**:
  * Calculate threshold: $threshold = \max(\frac{avg\_notehead\_width}{2}, 15.0)$ pixels
  * If $|x_i - x_j| < threshold$ for two barlines, merge them by averaging their X coordinates
  * Result: Set of **Global Barline** X positions

#### Step 4: Create Measure Buckets
* Based on Global Barline X positions $(x_1, x_2, ..., x_n)$:
  * **First measure**: symbols where $x_{center} < x_1$
  * **Middle measures**: symbols where $x_i \le x_{center} < x_{i+1}$ for each consecutive pair
  * **Last measure**: symbols where $x_{center} \ge x_n$
* Filter out empty measures (containing only barlines)

#### Fallback: Spacing-Based Inference
If no barlines detected:
* Calculate average symbol spacing
* Detect large gaps (gap $> 3 \times avg\_gap$) as measure boundaries

---

## Phase 3: Symbol Assembly (Note Construction)

**Goal**: Link symbols (noteheads, stems, beams, flags, dots, accidentals) into complete note objects.

### Rule 4: Notehead-to-Stem Linking

For each **notehead** symbol:

#### Case 1: Filled Noteheads (notehead-full)
* **Priority 1 - Overlapping Stem**:
  * Search for stems where bounding boxes overlap
  * If multiple overlapping stems found, choose the one with minimum distance to notehead center
  * Link notehead $\rightarrow$ stem

* **Priority 2 - Virtual Stem Creation** (if no overlapping stem found):
  1. **Search for Beam/Flag**:
     * Search direction: both upward and downward from notehead
     * Search range: half the distance to adjacent staff system (or $2 \times avg\_spacing$ as default)
     * Horizontal alignment: within $notehead.width \times 2$
     * Non-overlapping: beam/flag must not overlap with notehead ($|y_{beam} - y_{notehead}| > notehead.height / 2$)
  
  2. **Create Virtual Stem**:
     * If beams/flags found: extend virtual stem to reach all found beams/flags
     * If not found: create default-length virtual stem ($3.5 \times line\_spacing$)
     * **Stem Direction Correction**:
       * If notehead is at or below middle staff line: stem extends upward
       * If notehead is above middle staff line: stem extends downward
  
  3. **Virtual Stem Attributes**:
     * `class_name`: "stem"
     * `confidence`: 0.5 (indicates virtual)
     * `bbox`: calculated based on notehead position and found beams/flags

#### Case 2: Other Noteheads (notehead-half, notehead-whole)
* **Priority 1 - Overlapping Stem**: Same as filled noteheads
* **Priority 2 - Nearest Stem**: If no overlap, find the stem with minimum distance

### Rule 5: Stem-to-Beam/Flag Linking

For each **stem** (including virtual stems):

* **Search for Beams/Flags**:
  * Collect all overlapping beams/flags
  * Collect all non-overlapping beams/flags within $stem.width$ distance
  * **Distance Constraint**: Only consider beams/flags within one stem width
  * If no beam or flag found within this range, do not add any links

* **Priority**:
  * Overlapping beams/flags first (sorted by distance)
  * Then non-overlapping beams/flags (sorted by distance)

* **Multiple Linking**: Return ALL qualifying beams and flags, not just the first one

### Rule 6: Dot Detection

For each **notehead**:

* **Search Direction**: Rightward (positive X direction)
* **Geometric Constraints**:
  * Horizontal distance: $|dot.center_x - notehead.center_x| < DOT\_THRESHOLD_X$ (default: 30 pixels)
  * Vertical distance: $|dot.center_y - notehead.center_y| < DOT\_THRESHOLD_Y$ (default: 30 pixels)
* **Effect**: If dot found, multiply duration by 1.5 (dotted note)

### Rule 7: Accidental Detection

For each **notehead**:

* **Search Direction**: Leftward (negative X direction, typically)
* **Valid Accidentals**: `sharp`, `flat`, `natural`, `accidental`
* **Geometric Constraints**:
  * Horizontal distance: $|accidental.center_x - notehead.center_x| < ACCIDENTAL\_THRESHOLD_X$ (default: 40 pixels)
  * Vertical distance: $|accidental.center_y - notehead.center_y| < ACCIDENTAL\_THRESHOLD_Y$ (default: 50 pixels)
* **Effect**: Apply accidental to pitch calculation

---

## Phase 4: Attributes Recognition (Musical Context)

**Goal**: Determine clef, key signature, and time signature for each measure. These attributes exhibit **state persistence** (carry forward to subsequent measures).

### Rule 8: Clef Detection

For each measure $M_i$ in each staff $S_j$:

1. **Search**: Look for symbols with `class_name` containing: `g-clef`, `f-clef`, `c-clef`
2. **Assignment**:
   * If found: Update current staff's `Current_Clef` state
   * If not found:
     * **First measure** ($M_1$): Force assignment (default: Part 1 = Treble, Part 2 = Bass) with low confidence
     * **Subsequent measures**: Inherit clef from previous measure $M_{i-1}$

### Rule 9: Key Signature Detection

Key signatures typically appear after clef and before time signature.

#### Pattern A: Whole Label
* If `key_signature` label detected, use directly

#### Pattern B: Discrete Symbol Clustering
* **Search Region**: Right of clef, left of time signature or first note
* **Count Accidentals**: Count number of `sharp`, `flat`, or `natural` symbols
* **Clustering Logic**: If a group of sharps/flats has X-axis distance $< 1.5 \times note\_width$, treat as same key signature group
* **Semantic Parsing**:
  * 1 sharp $\rightarrow$ G Major / E Minor
  * 3 flats $\rightarrow$ Eb Major / C Minor
  * No accidentals $\rightarrow$ C Major / A Minor

#### State Update
* Found: Update current measure's key signature
* Not found: Inherit from previous measure

#### Key Signature Application Rule
Once a key signature is detected, it **persists and applies to all subsequent measures** until a new key signature is detected.

**Application Logic**:
1. **Convert Key Signature String to Note Mapping**:
   * Sharps follow circle of fifths order: F, C, G, D, A, E, B
     * "1#" → F sharp
     * "2#" → F, C sharp
     * "3#" → F, C, G sharp
   * Flats follow reverse circle of fifths order: B, E, A, D, G, C, F
     * "1b" → B flat
     * "2b" → B, E flat
     * "3b" → B, E, A flat

2. **Apply to Pitch Calculation**:
   * For each note in all subsequent measures:
     * Calculate base pitch from geometric position (ignoring accidentals)
     * If note has **local accidental** (temporary sharp/flat/natural): Apply local accidental (overrides key signature)
     * If note has **no local accidental**: Apply key signature accidental for that note name
     * Result: Pitch name includes accidental from key signature (e.g., "F#4" if key has F sharp)

3. **Priority Order**:
   * **Highest Priority**: Local accidental (temporary sharp/flat/natural) - overrides key signature
   * **Lower Priority**: Key signature accidental - applies if no local accidental present

### Rule 10: Time Signature Detection

1. **Search for**:
   * `time_signature` label
   * `letter_c` (Common Time = 4/4)
   * Vertical Digit Stack: e.g., `numeral_3` directly above `numeral_4` (indicating 3/4)

2. **State Update**:
   * Found: Update time signature
   * Not found: Inherit from previous measure

3. **Anomaly Detection**:
   * If $\sum note.duration > time\_signature\_duration + tolerance$ (measure overflow)
   * Log "Time Signature Mismatch" warning (unless it's a pickup measure)

---

## Phase 5: Tuplet Detection (Triplets & Other Groups)

**Goal**: Detect triplets and other tuplets using a cascaded rule set from strong to weak constraints.

### Definitions

* **Input**:
  * $N_{sorted}$: Notes sorted by X coordinate
  * $S_{tuplet}$: Tuplet marker symbols (`numeral_3`, `tuple_bracket`)
  * $B$: Beam objects

* **Search Region**: Typically extends 0.5-3 staff heights around the tuplet marker

### Rule Table: Triplet Detection Logic

#### Rule 1: Bracket-Based Detection [Highest Priority]

Most explicit indication, typically appears above/below groups of quarter notes without beams.

1. **Iterate**: All symbols where `class_name` contains `tuple_bracket` or `tuple_line`
2. **Spatial Projection**:
   * Get bracket X-axis range: $[x_{start}, x_{end}]$
   * Find all notes whose notehead center X falls within $[x_{start}, x_{end}]$
3. **Vertical Filtering**:
   * Remove notes whose vertical distance to bracket exceeds $1.5 \times staff\_height$
4. **Numeral Association**:
   * Search for `numeral_3` within $0.5 \times staff\_height$ radius of bracket center
   * **Relaxed Condition**: If no `numeral_3` found but exactly 3 notes are inside bracket $\rightarrow$ still recognize as triplet
5. **Confirmation**: If exactly 3 notes found, mark as triplet group

#### Rule 2: Beam-Based Detection [Second Priority]

Most common case for eighth notes or sixteenth notes.

1. **Iterate**: All `beam` objects
2. **Filter**: Find beams linking 3 or multiples of 3 stems
3. **Exclude**: Skip if time signature is 6/8, 9/8, or 12/8 (compound meters)
4. **Numeral Search**:
   * Define search region: radius = $3 \times line\_spacing$ around beam center
   * Find `numeral_3` within search region
   * **Collision Check**: `numeral_3` should overlap beam or be directly above/below beam
5. **Finger Number Exclusion**:
   * If `numeral_3` is strictly vertically aligned above a single notehead (not centered over the group), likely a finger number $\rightarrow$ exclude
   * Triplet markers typically appear at the **geometric center X** of the note group
6. **Confirmation**: If conditions met, mark all notes under this beam as triplet

#### Rule 3: Numeral-3-Centered Detection [Lowest Priority]

Handles isolated "3" symbols without brackets or beams (common in dense typesetting).

1. **Iterate**: All unused `numeral_3` symbols (excluding those in time signatures)
2. **Time Signature Exclusion**:
   * Check if `numeral_3` is part of a vertical digit stack (e.g., 3/4, 3/8)
   * If vertically aligned with another numeral ($|x_{diff}| < 0.8 \times spacing$) and forms valid time signature pattern, exclude
3. **Distance-Based Search**:
   * Calculate Euclidean distance from `numeral_3` to all notes in measure
   * Search radius: $6 \times line\_spacing$
   * Select **3 closest notes**
4. **Finger Number Exclusion**:
   * If `numeral_3` is directly above/below a single note (vertical alignment + close distance), treat as finger number $\rightarrow$ exclude
5. **Confirmation**: If 3 unprocessed notes found, infer as triplet

#### Rule 4: Tuplet Modification Attributes

Once a note group $G = \{n_1, n_2, n_3\}$ is identified as a triplet:

1. **Duration Modification**:
   * For each $n \in G$:
     * `time_modification_actual_notes` = 3
     * `time_modification_normal_notes` = 2
     * `duration_xml` = `base_duration` $\times \frac{2}{3}$

2. **MusicXML Tuplet Tags**:
   * $n_1$ (first note): Add `<tuplet type="start" bracket="yes/no"/>`
   * $n_2$ (middle note): No tuplet tag (or optional `<tuplet type="continue"/>`)
   * $n_3$ (last note): Add `<tuplet type="stop"/>`

3. **Bracket Display Logic**:
   * **Rule 1** (Bracket-triggered): `bracket="yes"`
   * **Rule 2** (Beam-triggered): `bracket="no"` (beam already visible, bracket redundant)
   * **Rule 3** (Loose numeral): `bracket="yes"` (for clarity, force bracket in output)

#### Rule 5: Sanity Check (Fallback) [Last Resort]

Detects implicit triplets based on measure duration overflow.

1. **Trigger Condition**: $\sum note.duration > time\_signature\_duration + tolerance$
2. **Search for Correction**:
   * Iterate through all unprocessed consecutive 3-note groups $\{n_i, n_{i+1}, n_{i+2}\}$
   * **Condition A (Uniform Duration)**: All 3 notes must have identical `base_duration`
   * **Condition B (Fixes Overflow)**: Converting to triplet reduces total duration by $base\_duration$
   * **Condition C (No Numeral Conflict)**: No `numeral_3` detected in this note group's region
3. **Confirmation**: If all conditions satisfied, force conversion to triplet with `confidence="Low"` and `tuplet_rule_triggered="Rule5"`

---

## Phase 6: Pitch Determination

**Goal**: Calculate pitch names (e.g., "C4", "F#5") from notehead vertical positions.

### Rule 11: Pitch Calculation Algorithm

#### Reference Standard

* Use **Top Line** (highest line of the staff) as the reference (Step 0)
* Reference pitches for each clef type:
  * **Treble Clef (G-Clef)**: Top Line = **F5**
  * **Bass Clef (F-Clef)**: Top Line = **A3**
  * **Alto Clef (C-Clef)**: Top Line = **G4**

#### Step Calculation

1. **Find Closest Staff Line**:
   * For notehead at position $y_{note}$, find the staff line $y_{closest}$ with minimum $|y_{note} - y_{closest}|$
   * Determine the line index $i_{closest}$ (top line = 0, next line = 1, ...)

2. **Calculate Base Steps**:
   * Each staff line = 2 steps apart
   * Base steps from top line: $steps_{base} = i_{closest} \times 2$

3. **Calculate Local Offset**:
   * Half-spacing: $h = \frac{median(line\_spacings)}{2}$
   * Local offset: $steps_{local} = \frac{y_{note} - y_{closest}}{h}$
   * Total steps: $steps_{total} = steps_{base} + steps_{local}$

4. **Rounding**: Round to nearest 0.5 step (half-step precision)

#### Pitch Derivation

* Map steps to diatonic scale: $Scale = [C, D, E, F, G, A, B]$
* Starting from reference pitch (e.g., F5 for Treble Clef), move down $steps_{total}$ diatonic steps
* Formula:
  * Absolute pitch index: $abs_{ref} = ref\_octave \times 7 + ref\_note\_index$
  * Target pitch index: $abs_{target} = abs_{ref} - steps_{total}$
  * Target octave: $target\_octave = \lfloor abs_{target} / 7 \rfloor$
  * Target note: $target\_note = Scale[abs_{target} \mod 7]$

#### Accidental Application

1. **Calculate Base Pitch**: First, determine visual natural pitch (ignoring accidentals)
2. **Priority Check**: Check if notehead has a **local accidental** (temporary sharp/flat/natural symbol)
3. **Apply Accidentals** (in priority order):
   * **If local accidental exists**: Apply local accidental (overrides key signature)
   * **If no local accidental**: Apply key signature's accidental for this note name (if key signature is active)
4. **Key Signature Persistence**: Once a key signature is detected, it applies to **all subsequent measures** until changed
   * Key signature affects all notes with the same note name (e.g., all F notes become F# if key has F sharp)
   * Key signature persists across measure boundaries automatically
5. **Final Pitch**: Combine note name, accidental, and octave (e.g., "F#4", "Bb3")

---

## Phase 7: Global Indexing & Assembly

### Rule 12: Global Measure Numbering

MusicXML requires continuous measure numbering across all systems.

* Initialize `Global_Index = 1`
* Traverse systems in order ($Sys_1 \rightarrow Sys_N$):
  * For each measure $M$ in the system (left to right):
    * If not a **multi-measure rest**:
      * $M.number = Global\_Index$
      * $Global\_Index += 1$
    * If `multi-measure_rest` detected (with count $N$):
      * $M.number = Global\_Index$
      * $Global\_Index += N$

### Rule 13: Pickup Measure (Anacrusis) Handling

* **Condition**: First measure of the score ($Measure_1$)
* **Check**: Calculate total duration $D_{sum} = \sum note.duration$
* **Logic**: If $D_{sum} < time\_signature\_duration$:
  * Mark measure as `implicit="yes"` (MusicXML attribute)
  * Do not report as error (valid pickup measure)

---

## Output Rules

* **Test Output Directory**: All test outputs saved to `Output/test`
* **Visualization Files**: Saved to `Output/test` with debug annotations:
  * Linked pairs drawn as arrows/lines
  * Triplet groups highlighted with colored rectangles
  * Rule triggers annotated (e.g., "Triplet (Rule 1: Bracket)")

---

## Algorithm Summary

This pipeline transforms raw symbol detections into a structured musical score through six major phases:

1. **System & Staff Grouping**: Organize staff lines into musical rows and voice parts
2. **Measure Slicing**: Segment time horizontally using barlines
3. **Symbol Assembly**: Link noteheads, stems, beams, flags, dots, and accidentals
4. **Attribute Recognition**: Detect clef, key signature, and time signature with state persistence
5. **Tuplet Detection**: Apply cascaded rules (Bracket $\rightarrow$ Beam $\rightarrow$ Numeral $\rightarrow$ Sanity Check)
6. **Pitch Determination**: Convert geometric positions to musical pitches using clef-aware calculation

All rules are designed with **priority ordering** and **fallback mechanisms** to handle various notation styles and detection errors gracefully.
