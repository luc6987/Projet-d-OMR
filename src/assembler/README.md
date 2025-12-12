# OMR Assembly Module

This module is responsible for the **Semantic Reconstruction** phase of the Optical Music Recognition pipeline.
It takes the output from the visual perception models (U-Net for staff removal, YOLO for symbol detection) and reconstructs the musical score.

## Components

1.  **Staff Detection (`staff.py`)**:
    *   Input: Binary/Grayscale Mask from U-Net (where staff lines are marked).
    *   Process: Horizontal projection and peak finding to locate staff lines.
    *   Output: Grouped `StaffSystem` objects (sets of 5 lines).

2.  **Symbol Loading (`symbols.py`)**:
    *   Input: JSON output from YOLO detection.
    *   Process: Filtering by confidence, creating `Symbol` objects.
    *   Output: List of `Symbol` objects with bounding box info.

3.  **Music Theory Logic (`theory.py`)**:
    *   Process: Mapping geometric Y-coordinates to musical pitches based on staff line positions and Clefs.
    *   Output: Pitch strings (e.g., "C4", "F#5").

4.  **Score Builder (`builder.py`)**:
    *   Process:
        *   Assigns symbols to nearest staff systems.
        *   Sorts symbols by time (X-axis).
        *   Detects Clefs and Barlines.
        *   Associates Accidentals with Notes.
        *   Constructs Measures and Notes.
    *   Output: Structured `AssembledPart` objects.

5.  **Exporter (`exporter.py`)**:
    *   Process: Converts internal objects to `music21` objects.
    *   Output: MusicXML file (.xml).

## Usage

Run the main script via module entry:

```bash
python src/main.py infer assembler
# Or directly:
python src/assembler/run_assembly.py \
    --json path/to/yolo_results.json \
    --mask path/to/unet_mask.png \
    --output path/to/output.xml
```

## Dependencies

*   `music21`: For MusicXML generation.
*   `opencv-python`: For image processing.
*   `scipy`: For signal processing (peak finding).
*   `numpy`: For numerical operations.

