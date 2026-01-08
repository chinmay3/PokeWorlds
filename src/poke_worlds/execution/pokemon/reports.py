from poke_worlds.execution.report import ExecutionReport
from poke_worlds.emulation.pokemon.trackers import PokemonOCRTracker
from typing import Dict, Any

class PokemonExecutionReport(ExecutionReport):
    REQUIRED_STATE_TRACKER = PokemonOCRTracker

    def state_info_to_str(self, state_info: Dict[str, Dict[str, Any]]) -> str:
        # just get the OCR text info:
        if "ocr" in state_info and "transition_ocr_texts" in state_info["ocr"]:
            ocr_texts = state_info["ocr"]["transition_ocr_texts"]
            return "OCR Info: " + "\n".join([f"{k}: {v}" for k, v in ocr_texts.items()])
        else:
            return ""