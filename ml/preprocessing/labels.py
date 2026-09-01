"""
NIH ChestX-ray14 Pathology Labels & Binary Indicator Utilities
--------------------------------------------------------------
Defines the official 14 NIH pathology classes, handles 'No Finding',
and parses finding label strings into 14-dimensional target vectors.
"""

from typing import List, Union
import numpy as np

# Official 14 NIH Pathology Classes (Preserving exact spelling)
PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]

NO_FINDING_LABEL = "No Finding"
NUM_CLASSES = len(PATHOLOGY_CLASSES)


def parse_finding_labels_to_vector(labels_str: str) -> np.ndarray:
    """
    Parses an NIH Finding Labels string into a 14-element binary numpy array.

    Args:
        labels_str: Pipe-delimited string (e.g. 'Atelectasis|Effusion' or 'No Finding')

    Returns:
        np.ndarray: 14-dimensional float32 vector containing 0.0 or 1.0.
    """
    target_vector = np.zeros(NUM_CLASSES, dtype=np.float32)

    if not labels_str or str(labels_str).strip() == NO_FINDING_LABEL:
        return target_vector

    tokens = [t.strip() for t in str(labels_str).split("|")]
    for token in tokens:
        if token in PATHOLOGY_CLASSES:
            idx = PATHOLOGY_CLASSES.index(token)
            target_vector[idx] = 1.0

    return target_vector
