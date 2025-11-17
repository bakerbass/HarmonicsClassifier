"""
XML parser module for extracting note annotations from IDMT-SMT-GUITAR XML files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_xml_events(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse XML annotation file and extract all note events.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of dictionaries containing note event data
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}")
        return []
    
    events = []
    
    for ev in root.findall(".//event"):
        event_data = _extract_event_data(ev)
        if event_data:
            events.append(event_data)
    
    return events


def _extract_event_data(event_element: ET.Element) -> Optional[Dict[str, Any]]:
    """
    Extract data from a single event XML element.
    
    Args:
        event_element: XML element representing a note event
        
    Returns:
        Dictionary with event data or None if required fields are missing
    """
    def get_text(tag: str) -> Optional[str]:
        """Helper to safely get text from XML element."""
        node = event_element.find(tag)
        return node.text.strip() if node is not None and node.text else None
    
    # Extract required fields
    onset_str = get_text("onsetSec")
    offset_str = get_text("offsetSec")
    pitch_str = get_text("pitch")
    
    # Skip events missing critical data
    if not all([onset_str, offset_str, pitch_str]):
        return None
    
    try:
        onset = float(onset_str)
        offset = float(offset_str)
        pitch = int(pitch_str)
    except (ValueError, TypeError):
        return None
    
    # Build event dictionary
    event_data = {
        "onset_sec": onset,
        "offset_sec": offset,
        "duration_sec": offset - onset,
        "pitch_midi": pitch,
        "excitation_style": get_text("excitationStyle"),
        "expression_style": get_text("expressionStyle"),
        "string_number": get_text("stringNumber"),
        "fret_number": get_text("fretNumber"),
    }
    
    return event_data


def validate_events(events: List[Dict[str, Any]], min_duration: float = 0.01) -> List[Dict[str, Any]]:
    """
    Validate and filter events based on quality criteria.
    
    Args:
        events: List of event dictionaries
        min_duration: Minimum duration threshold in seconds
        
    Returns:
        Filtered list of valid events
    """
    valid_events = []
    
    for event in events:
        # Check duration
        if event["duration_sec"] < min_duration:
            continue
        
        # Check for negative duration
        if event["onset_sec"] >= event["offset_sec"]:
            continue
        
        valid_events.append(event)
    
    return valid_events


def get_event_statistics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about parsed events.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Dictionary with event statistics
    """
    if not events:
        return {"total_events": 0}
    
    expression_styles = {}
    excitation_styles = {}
    
    for event in events:
        expr = event.get("expression_style", "unknown")
        excit = event.get("excitation_style", "unknown")
        
        expression_styles[expr] = expression_styles.get(expr, 0) + 1
        excitation_styles[excit] = excitation_styles.get(excit, 0) + 1
    
    durations = [e["duration_sec"] for e in events]
    
    return {
        "total_events": len(events),
        "expression_styles": expression_styles,
        "excitation_styles": excitation_styles,
        "avg_duration_sec": sum(durations) / len(durations),
        "min_duration_sec": min(durations),
        "max_duration_sec": max(durations),
    }
