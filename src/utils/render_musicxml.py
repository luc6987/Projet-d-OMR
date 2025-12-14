
import os
import sys
import argparse
import verovio

def render_musicxml(input_path, output_path=None):
    """
    Render a MusicXML file to an SVG image using Verovio.
    
    Args:
        input_path (str): Path to the input MusicXML file.
        output_path (str): Path to save the output SVG file. If None, it will be derived from input path.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False

    # Initialize Verovio toolkit
    tk = verovio.toolkit()
    
    # Load the MusicXML file
    try:
        tk.loadFile(input_path)
    except Exception as e:
        print(f"Error loading MusicXML file: {e}")
        return False

    # Set rendering options
    # adjustPageHeight: true ensures the page expands to fit all systems
    # pageWidth: Set reasonably distinct width
    tk.setOptions({
        "pageWidth": 2100,
        "adjustPageHeight": 'true',
        "scale": 50, # Adjust scale if needed
        "header": 'none',
        "footer": 'none'
    })

    # Render the first page to SVG
    # With adjustPageHeight, everything should remain on page 1
    try:
        svg_data = tk.renderToSVG(1) # Render page 1
    except Exception as e:
        print(f"Error rendering to SVG: {e}")
        return False

    # Determine output path if not provided
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".svg"

    # Add white background if it's an SVG
    if output_path.lower().endswith('.svg'):
        # Find the end of the opening <svg> tag
        svg_start_idx = svg_data.find('<svg')
        if svg_start_idx != -1:
            close_bracket_idx = svg_data.find('>', svg_start_idx)
            if close_bracket_idx != -1:
                # Insert a white rectangle covering the whole area
                # We interpret "100%" width/height as covering the viewport
                white_rect = '\n    <rect width="100%" height="100%" fill="white"/>'
                svg_data = svg_data[:close_bracket_idx+1] + white_rect + svg_data[close_bracket_idx+1:]

    # Save the SVG data
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_data)
        print(f"Successfully rendered {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving output file: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render MusicXML to SVG using Verovio")
    parser.add_argument("input_path", help="Path to the input MusicXML file")
    parser.add_argument("--output", "-o", help="Path to the output SVG file", default=None)
    
    args = parser.parse_args()
    
    success = render_musicxml(args.input_path, args.output)
    sys.exit(0 if success else 1)
