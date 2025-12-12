# Assembly Visualization

When running the assembly pipeline with the `--visualize` flag, a visualization folder will be created containing:

## Generated Files

1. **00_summary.json** - Statistics about symbols, links, and staff systems
2. **01_original.png** - Original input image
3. **02_unet_mask.png** - U-Net staff line detection mask
4. **03_unet_cleaned.png** - U-Net cleaned image (staff lines removed)
5. **04_yolo_detections.jpg** - YOLO symbol detection results with bounding boxes
6. **05_assembled_links.jpg** - Assembled symbol relationships with connecting lines

## Usage

```bash
python src/main.py infer assembler
# Or directly:
python src/assembler/run_assembly.py \
    --json Output/inference_output/p001_results.json \
    --mask Output/UNet/w-01/p001_mask.png \
    --output Output/p001.xml \
    --visualize
```

## Visualization Details

### 04_yolo_detections.jpg
- Shows all detected symbols with colored bounding boxes
- Color coding:
  - Green: Noteheads
  - Blue: Stems
  - Red: Beams
  - Cyan: Flags
  - Magenta: Clefs
  - Yellow: Rests
  - Purple: Accidentals
  - Orange: Barlines

### 05_assembled_links.jpg
- Shows symbol relationships found during assembly
- Lines connect linked symbols (e.g., notehead-stem, notehead-beam)
- Color coding for link types:
  - Cyan: Stem links
  - Magenta: Beam links
  - Yellow: Flag links
  - Green: Dot links
  - Blue: Accidental links
- Gray lines show detected staff systems
- All symbols shown with light gray boxes


