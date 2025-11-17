"""
Labeling module for deriving classification labels from expression styles.
"""

from typing import Dict, Any
from . import config


def derive_labels(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive classification labels from expression style.
    
    Adds the following fields to the event dictionary:
    - is_harmonic: Boolean flag for natural harmonics
    - is_dead_note: Boolean flag for dead notes
    - is_general_note: Boolean flag for general notes
    - label_category: String category (harmonic, dead_note, general_note)
    
    Args:
        event: Event dictionary from XML parser
        
    Returns:
        Event dictionary with added label fields
    """
    expr = event.get("expression_style", "")
    
    # Determine label flags
    is_harm = expr == config.EXPR_HARMONIC
    is_dead = expr == config.EXPR_DEAD_NOTE
    is_gen = not is_harm and not is_dead
    
    # Determine category
    if is_harm:
        category = config.LABEL_HARMONIC
    elif is_dead:
        category = config.LABEL_DEAD_NOTE
    else:
        category = config.LABEL_GENERAL_NOTE
    
    # Update event with labels
    event.update({
        "is_harmonic": is_harm,
        "is_dead_note": is_dead,
        "is_general_note": is_gen,
        "label_category": category,
    })
    
    return event


def filter_by_label(events: list, label_category: str) -> list:
    """
    Filter events by label category.
    
    Args:
        events: List of labeled event dictionaries
        label_category: Category to filter by (harmonic, dead_note, general_note)
        
    Returns:
        Filtered list of events
    """
    return [e for e in events if e.get("label_category") == label_category]


def get_label_distribution(events: list) -> Dict[str, int]:
    """
    Get distribution of labels across all events.
    
    Args:
        events: List of labeled event dictionaries
        
    Returns:
        Dictionary mapping label categories to counts
    """
    distribution = {
        config.LABEL_HARMONIC: 0,
        config.LABEL_DEAD_NOTE: 0,
        config.LABEL_GENERAL_NOTE: 0,
    }
    
    for event in events:
        category = event.get("label_category")
        if category in distribution:
            distribution[category] += 1
    
    return distribution
