from poke_worlds.emulation.parser import StateParser

class LegendOfZeldaParser(StateParser):
    def __init__(self, pyboy, parameters):
        self.rom_data_path = parameters["legend_of_zelda_rom_data_path"]
        super().__init__(pyboy, parameters)

    def __repr__(self) -> str:
        return "LegendOfZeldaParser"

