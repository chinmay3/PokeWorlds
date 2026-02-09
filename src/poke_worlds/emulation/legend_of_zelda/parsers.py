from poke_worlds.utils import verify_parameters, log_error
from poke_worlds.emulation.parser import StateParser


class LegendOfZeldaParser(StateParser):
    """
    Minimal state parser for Legend of Zelda: Link's Awakening.

    This parser only provides the ROM data path required by the base StateParser.
    No custom screen regions or metrics are defined.
    """

    def __init__(self, pyboy, parameters):
        """
        Args:
            pyboy: PyBoy emulator instance.
            parameters: Project parameters loaded from configs.
        """
        verify_parameters(parameters)
        variant = "legend_of_zelda"
        if f"{variant}_rom_data_path" not in parameters:
            log_error(
                f"ROM data path not found for variant: {variant}. Add {variant}_rom_data_path to the config files.",
                parameters,
            )
        self.rom_data_path = parameters[f"{variant}_rom_data_path"]
        super().__init__(pyboy, parameters)

    def __repr__(self) -> str:
        return "LegendOfZeldaParser"
