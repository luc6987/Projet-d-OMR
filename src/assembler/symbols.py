import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Symbol:
    """
    Represents a detected symbol from YOLO.
    """
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    
    @property
    def x1(self) -> float: return self.bbox[0]
    @property
    def y1(self) -> float: return self.bbox[1]
    @property
    def x2(self) -> float: return self.bbox[2]
    @property
    def y2(self) -> float: return self.bbox[3]
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
        
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
        
    @property
    def height(self) -> float:
        return self.y2 - self.y1

class SymbolLoader:
    """
    Loads and filters symbols from YOLO JSON output.
    """
    def __init__(self, json_path: Path, min_confidence: float = 0.25):
        self.json_path = json_path
        self.min_confidence = min_confidence
        self.symbols: List[Symbol] = []
        
    def load(self) -> List[Symbol]:
        """
        Loads symbols from JSON file.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Data can be a list (if direct list of dicts) or dict (if wrapped)
        # Based on user's script output, it seems to be a list of dicts
        if isinstance(data, dict) and 'detections' in data:
             raw_detections = data['detections']
        elif isinstance(data, list):
             raw_detections = data
        else:
             print(f"[SymbolLoader] Warning: Unexpected JSON format in {self.json_path}")
             return []
             
        self.symbols = []
        for det in raw_detections:
            # Check format
            if 'confidence' not in det or 'bbox' not in det:
                continue
                
            if det['confidence'] < self.min_confidence:
                continue
                
            symbol = Symbol(
                class_name=det.get('class_name', 'unknown'),
                confidence=det['confidence'],
                bbox=det['bbox']
            )
            self.symbols.append(symbol)
            
        print(f"[SymbolLoader] Loaded {len(self.symbols)} symbols (conf >= {self.min_confidence}).")
        return self.symbols
        
    def sort_by_time(self, symbols: List[Symbol]) -> List[Symbol]:
        """
        Sorts symbols by x-coordinate (time).
        """
        return sorted(symbols, key=lambda s: s.center_x)

