<p align="center">
  <picture>
    <img alt="Pokémon Environments" src="assets/blackicon.png"   width="500" height="400" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<h2 align="center">
    <p>Gym Environments and Testbeds for Pokémon</p>
</h2>

<p align="center">
    <a href="https://github.com/DhananjayAshok/PokemonEnvironments/blob/main/LICENSE" target="_blank" rel="noopener noreferrer"><img alt="GitHub" src="https://img.shields.io/badge/license-MIT-blue"></a>
    <a href="https://dhananjayashok.github.io/" target="_blank" rel="noopener noreferrer"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://dhananjayashok.github.io/PokemonEnvironments/" target="_blank" rel="noopener noreferrer"><img alt="GitHub" src="https://img.shields.io/badge/documentation-pdoc-red"></a>
</p>



This repo is under development

## Installation

The installation consists of four steps:
1. Environment Setup
2. ROM Setup
3. (Optional) Storage Directory Configuration
4. Final test

#### Environment Setup
Create and activate a virtual environment with [uv](https://docs.astral.sh/uv/), a fast Rust-based Python package and project manager.



```bash
uv venv /path/to/env --python=3.12
```
Then, source the environment
* On Windows:
```powershell
/path/to/env/Scripts/Activate
```
* On Linux:
```bash
source /path/to/env/bin/activate
```

Then, clone the repo and set up dependencies:
```
git clone https://github.com/DhananjayAshok/PokemonEnvironments
cd PokemonEnvironments
uv sync --active
```
#### ROM Setup

Next, you must legally acquire ROMs for Pokémon from Nintendo (perhaps by dumping the ROM file from your own catridge). We discourage any attempts to use this repository with unofficialy downloaded ROMs. The following base game ROMs are supported by this repository:
* Pokémon Red (save as `PokemonRed.gb`)
* Pokémon Crystal (save as `PokemonCrystal.gbc`)

Additionally, our testing environment uses several Pokémon ROM patches / hacks that alter the game in some way. These can be obtained by acquiring the original ROM and applying a "patch" to it. After patching the original ROM, you will be left with a `.gb` or `.gbc` file. We support:
* [Pokémon Brown](https://rainbowdevs.com/title/brown/) (save as `PokemonBrown.gbc`)
* [Pokémon Prism](https://rainbowdevs.com/title/prism/) (save as `PokemonPrism.gbc`)
* [Pokémon Quarantine Crystal](https://www.pokecommunity.com/threads/quarantinecrystal-full-fakedex-12-gym-demo-out-now-v-0-804-updated-03-02-2024.436807/) (save as `PokemonQuarantineCrystal.gbc`)
* [Pokémon Fool's Gold](https://www.pokecommunity.com/threads/pok%C3%A9mon-fools-gold-a-hack-of-crystal-where-everything-is-familiar-yet-different.433723/) (save as `PokemonFoolsGold.gbc`)
* [Pokémon Star Beasts](https://www.pokecommunity.com/threads/star-beasts-asteroid-version.530552/) (save as `PokemonStarBeasts.gb`)

Once you have a ROM (`.gb` or `.gbc` file), place it in the appropriate path. For example, the ROM for Pokémon Red should be placed in `PokemonEnvironments/rom_data/pokemon_red/PokemonRed.gb`. See the [config folder](configs/) for the expected path to each supported game. 



#### Storage Directory Configuration
By default, this project assumes that you can store emulator outputs (logs, screenshots, video captures etc.) in the `storage` folder under the root directory of the project. If you wish to set a different location for storage, edit the appropriate configuration setting in the [config file](configs/private_vars.yaml).

#### Test

Check that setup is fine by running:
```bash
python demo.py
```
This should open up a GameBoy window where you can play the Pokemon Red game. 

To see how a random agent does, try:
```bash
python demo.py --play_mode random
```


## Quickstart


## Core Features


## Development
You may want to create your own initial states to start the emulation from. 

First, start with mGBA and save the game in the state you want to restore from. This will make a ROMNAME.sav file. Then run:

```bash
python dev/gameboy/save_state.py
```

This will save the state to `tmp.state`

Alternatively, run the pyboy simulator with:
```bash 
python dev/dev_play.py
```

This will run the game with the option to enter dev mode. If you want to save a particular state, go to [the gameboy configs](configs/gameboy_vars.yaml) *while* playing the game (at the state you want to save), change the `gameboy_dev_play_stop` parameter to `true` and then check the terminal. You will get a message with the possible dev actions. 

* `s`: saves the state (useful for creating fixed starting points)
* `c`: captures the screen in a particular [named region](src/poke_env/emulators/emulator.py) at the current game frame (useful for creating reference images of particular states e.g. menu/pc open, dialogues that trigger on particular milestones or events etc.) WRITE A README WITH EXAMPLES ON BOTH OF THESE


### Adding a new ROM Hack
1. Create a `$variant_rom_data_path` parameter in the [configs](configs) (either as a new file or in an existing one, see [Pokémon Brown](configs/pokemon_brown_vars.yaml) for an example)
2. Obtain the ROM hack and place it in the desired path under the [ROM data folder](rom_data). 
3. Go to the [parsers](src/poke_env/emulators/pokemon/parsers.py) and add the required ROM hack. See the `PokemonBrownGameStateParser` as an example. 
4. Go to the [registry](src/poke_env/emulators/pokemon/__init__.py) and add the ROM hack to `VARIANT_TO_GB_NAME`, `_VARIANT_TO_BASE_MAP`, `_VARIANT_TO_PARSER`
5. Run `python dev/create_first_state.py --variant <variant_name>`. This will create a default state. You will not be able to run the `Emulator` on this ROM before doing this. 
6. Run `python dev/dev_play.py` (with the [`gameboy_dev_play_stop` parameter](configs/gameboy_vars.yaml) set to `false`) and proceed through the game until you reach a satisfactory default starting state. Then, open the [config file](configs/gameboy_vars.yaml) and set `gameboy_dev_play_stop` to `true` and save the file. This will trigger a dev mode and ask you for a terminal input. Enter `s <rom_data_path>/states/default.state` and you will set that as the new default state.

#### Capturing Screens
This repo uses screen captures and comparison of screen renders to determine state (e.g. menu open, in battle). In Pokémon, the screen markers occur in regular places, and the ROM hacks don't change this much either, making it a reliable way to check for events / flags. For the basic regions, run in dev play mode, stop the game at the flag and run `c <region_name>` to save the screen region at that point. The exact screens vary with the base game

##### Common:
* `dialogue_bottom_right`: Speak to someone or interact with an object. When the popup appears, capture. 
* `menu_top_right`: Open the menu and capture. 
* `battle_enemy_hp_text`/`battle_player_hp_text`: Get into a battle and capture 
* `dialogue_choice_bottom_right`: Get into a situation where the dialogue asks you to choose between options (e.g. confirmation of Yes/No when picking starter Pokémon). Capture that. 

#### Known Bugs / Missing Features:
* the map screenshot for crystal assumes a Jhoto map. Must do a similar process for Kanto. To add Kanto we should add another named screen region called map_bottom_right_kanto with same boundary as player_card_middle and then recapture it. 

##### Pokémon Red Base:

##### Pokémon Crystal Base:

## Citation

```bibtex
```
