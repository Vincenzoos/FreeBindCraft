"""AlphaFold 3 structure/PAE filename pairing validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Af3PairInfo:
    complex_id: str
    model_index: int


def parse_af3_structure_pairing(structure_file: str | Path) -> Af3PairInfo | None:
    match = re.match(r"^(?P<complex>.+)_model_(?P<index>\d+)$", Path(structure_file).stem)
    if not match:
        return None
    return Af3PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def parse_af3_pae_pairing(pae_file: str | Path) -> Af3PairInfo | None:
    match = re.match(r"^(?P<complex>.+)_full_data_(?P<index>\d+)$", Path(pae_file).stem)
    if not match:
        return None
    return Af3PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def validate_structure_pae_pairing(structure_file: str | Path, pae_file: str | Path) -> None:
    """
    Validate known structure/PAE naming schemes before running ipSAE.

    Today this enforces AlphaFold 3 filename pairing and same-folder placement.
    AF2/Boltz hooks are intentionally left as placeholders for future extension
    when needed.
    """
    structure_path = Path(structure_file)
    pae_path = Path(pae_file)
    structure_name = structure_path.name
    pae_name = pae_path.name

    af3_structure = parse_af3_structure_pairing(structure_file)
    af3_pae = parse_af3_pae_pairing(pae_file)
    if af3_structure or af3_pae:
        if not af3_structure or not af3_pae:
            raise ValueError(
                "AF3 structure/PAE mismatch: expected a structure like '*_model_N.cif' "
                f"paired with a PAE file like '*_full_data_N.json'. Got '{structure_name}' "
                f"and '{pae_name}'."
            )
        if af3_structure.complex_id != af3_pae.complex_id:
            raise ValueError(
                "AF3 structure/PAE mismatch: structure and PAE appear to come from "
                f"different complexes ('{structure_name}' vs '{pae_name}')."
            )
        if af3_structure.model_index != af3_pae.model_index:
            raise ValueError(
                "AF3 structure/PAE mismatch: structure and PAE refer to different model "
                f"indices for the same complex ('{structure_name}' vs '{pae_name}')."
            )
        if structure_path.resolve().parent != pae_path.resolve().parent:
            raise ValueError(
                "AF3 structure/PAE mismatch: structure and PAE must share the same parent folder. "
                f"Got '{structure_path.resolve().parent}' and '{pae_path.resolve().parent}'."
            )
        return

    # Placeholder for future AF2 pairing validation.
    # Placeholder for future Boltz pairing validation.
