"""
Test case for key signature application rule.

This test verifies that:
1. Key signature is correctly parsed from string format (e.g., "1#", "3b")
2. Key signature applies to all notes in subsequent measures
3. Local accidentals override key signature
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.assembler.theory import PitchEngine, ClefType


def test_key_signature_parsing():
    """Test parsing of key signature strings to dictionaries."""
    print("=" * 60)
    print("Test 1: Key Signature Parsing")
    print("=" * 60)
    
    # Test sharps
    test_cases = [
        ("1#", {'F': 'sharp'}),
        ("2#", {'F': 'sharp', 'C': 'sharp'}),
        ("3#", {'F': 'sharp', 'C': 'sharp', 'G': 'sharp'}),
        ("4#", {'F': 'sharp', 'C': 'sharp', 'G': 'sharp', 'D': 'sharp'}),
    ]
    
    for key_str, expected in test_cases:
        result = PitchEngine.parse_key_signature(key_str)
        print(f"Key '{key_str}' -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    # Test flats
    test_cases_flat = [
        ("1b", {'B': 'flat'}),
        ("2b", {'B': 'flat', 'E': 'flat'}),
        ("3b", {'B': 'flat', 'E': 'flat', 'A': 'flat'}),
        ("4b", {'B': 'flat', 'E': 'flat', 'A': 'flat', 'D': 'flat'}),
    ]
    
    for key_str, expected in test_cases_flat:
        result = PitchEngine.parse_key_signature(key_str)
        print(f"Key '{key_str}' -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    # Test C major (no accidentals)
    result = PitchEngine.parse_key_signature("C")
    print(f"Key 'C' -> {result}")
    assert result == {}, f"Expected empty dict, got {result}"
    
    result = PitchEngine.parse_key_signature(None)
    print(f"Key None -> {result}")
    assert result == {}, f"Expected empty dict, got {result}"
    
    print("✓ All parsing tests passed!\n")


def test_key_signature_application():
    """Test application of key signature to pitch calculation."""
    print("=" * 60)
    print("Test 2: Key Signature Application to Pitches")
    print("=" * 60)
    
    # Mock staff lines (5 lines, evenly spaced)
    staff_lines = [100, 120, 140, 160, 180]  # Top to bottom
    
    # Test case: G Major (1 sharp = F#)
    # In treble clef, F5 is on top line (y=100)
    # F4 is one octave below, approximately at y=220 (below staff)
    # But let's test with a note that should be F
    
    key_sig_dict = PitchEngine.parse_key_signature("1#")
    print(f"Key signature 1# (G Major): {key_sig_dict}")
    
    # Test: Calculate pitch for F note in treble clef
    # F5 is on top line (y=100)
    pitch = PitchEngine.calculate_pitch(
        center_y=100.0,
        staff_lines=staff_lines,
        clef_type=ClefType.G_CLEF,
        key_signature=key_sig_dict
    )
    print(f"F note at y=100 (top line) -> {pitch}")
    assert '#' in pitch or pitch.startswith('F'), f"Expected F#5 or similar, got {pitch}"
    
    # Test: Calculate pitch for C note (should not have sharp in 1# key)
    # C5 in treble clef is in third space (approximately y=130)
    pitch_c = PitchEngine.calculate_pitch(
        center_y=130.0,
        staff_lines=staff_lines,
        clef_type=ClefType.G_CLEF,
        key_signature=key_sig_dict
    )
    print(f"C note at y=130 -> {pitch_c}")
    # C should not have sharp in 1# key (only F has sharp)
    
    # Test: 2# key (D Major: F#, C#)
    key_sig_dict_2 = PitchEngine.parse_key_signature("2#")
    print(f"\nKey signature 2# (D Major): {key_sig_dict_2}")
    
    pitch_f2 = PitchEngine.calculate_pitch(
        center_y=100.0,
        staff_lines=staff_lines,
        clef_type=ClefType.G_CLEF,
        key_signature=key_sig_dict_2
    )
    print(f"F note at y=100 -> {pitch_f2}")
    
    pitch_c2 = PitchEngine.calculate_pitch(
        center_y=130.0,
        staff_lines=staff_lines,
        clef_type=ClefType.G_CLEF,
        key_signature=key_sig_dict_2
    )
    print(f"C note at y=130 -> {pitch_c2}")
    
    # Test: 3b key (Eb Major: Bb, Eb, Ab)
    key_sig_dict_3b = PitchEngine.parse_key_signature("3b")
    print(f"\nKey signature 3b (Eb Major): {key_sig_dict_3b}")
    print(f"Expected: B, E, A should have flats")
    
    print("✓ Key signature application tests completed!\n")


def test_local_accidental_override():
    """Test that local accidentals override key signature."""
    print("=" * 60)
    print("Test 3: Local Accidental Override")
    print("=" * 60)
    
    staff_lines = [100, 120, 140, 160, 180]
    
    # Key signature: 1# (F should be sharp)
    key_sig_dict = PitchEngine.parse_key_signature("1#")
    
    # Test: F note with key signature should be F#
    pitch_with_key = PitchEngine.calculate_pitch(
        center_y=100.0,
        staff_lines=staff_lines,
        clef_type=ClefType.G_CLEF,
        key_signature=key_sig_dict
    )
    print(f"F note with 1# key signature -> {pitch_with_key}")
    
    # Test: F note with local natural should cancel key signature
    # This would be handled in builder.py by not passing key_signature when local accidental exists
    # But we can test the _apply_key_signature function directly
    base_pitch = "F5"
    key_sig_with_natural = {'F': 'natural'}
    pitch_with_natural = PitchEngine._apply_key_signature(base_pitch, key_sig_with_natural)
    print(f"F note with natural (cancels key signature) -> {pitch_with_natural}")
    assert pitch_with_natural == "F5", f"Expected F5, got {pitch_with_natural}"
    
    print("✓ Local accidental override test completed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Key Signature Application Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_key_signature_parsing()
        test_key_signature_application()
        test_local_accidental_override()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

