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

This may be a pre-existing environment for another project. Then, source the environment
* On Windows:
```powershell
/path/to/env/Scripts/Activate
```
* On Linux:
```bash
source /path/to/env/bin/activate
```

Then, clone the repo and install it as a `pip` package:
```
git clone https://github.com/DhananjayAshok/PokemonEnvironments
cd PokemonEnvironments
uv pip install -e .
```

You can now `import poke_env` from anywhere.
#### ROM Setup

Next, you must legally acquire ROMs for Pokémon from Nintendo (perhaps by dumping the ROM file from your own catridge). We discourage any attempts to use this repository with unofficialy downloaded ROMs. The following base game ROMs are supported by this repository:
* Pokémon Red (save as `PokemonRed.gb`)
* Pokémon Crystal (save as `PokemonCrystal.gbc`)

Additionally, our testing environment uses several Pokémon ROM patches / hacks that alter the game in some way. These can be obtained by acquiring the original ROM and applying a "patch" to it. After patching the original ROM, you will be left with a `.gb` or `.gbc` file. We support:
* [Pokémon Brown](https://rainbowdevs.com/title/brown/) (save as `PokemonBrown.gbc`)
* [Pokémon Prism](https://rainbowdevs.com/title/prism/) (save as `PokemonPrism.gbc`)
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
This section goes into details on how you would implement new features in this repo. 

### I want to create my own starting states
Easy. The only question is whether you want to save an mGBA state (perhaps you use cheats to lazily put the agent in a very specific state) or save a PyBoy state directly (i.e. you start from an existing state and play to the new state).

**From mGBA state:**

First, start with mGBA and save the game (go to the start menu and save) in the state you want to restore from. This will make a `variant_ROMNAME.sav` file in the same directory as the rom (which should be in rom_data/variant/). Then run:

```bash
python dev/save_state.py --variant <variant> --state_name <name>
```

This will save the state to `rom_data/<variant>/states/name.state` and allows you to load it by specifying it as a state name. 


To get to the state from PyBoy, first make sure the `gameboy_dev_play_stop` parameter is [configured](configs/gameboy_vars.yaml) to `false`. Then, run:
```bash 
python dev/dev_play.py --variant <variant> --init_state <optional_starting_state>
```

This will run the game with the option to enter dev mode. Play the game like you usually would, until you reach the state you want to save. Then, go to [the gameboy configs](configs/gameboy_vars.yaml) *while* playing the game (at the state you want to save), change the `gameboy_dev_play_stop` parameter to `true` (save the configs file) and then check the terminal. You will get a message with the possible dev actions. The one you're looking for is `s <name>`, which saves the state.

Regardless of how you did it, you can test that your state save worked with:
```bash
python demo.py --variant <variant> --init_state <name>
```

### I want to add a new ROM Hack
Setting up a new ROM Hack is an easy process that doesn't take more than 10 minutes once you've understood how. Please do reach out to me if you have any questions, and we can work to merge the new ROM into the repo together. 

#### Initial Steps:

0. Set the repo to `debug` mode by editing the [config file](configs/project_vars.yaml)
1. Create a `$variant_rom_data_path` parameter in the [configs](configs) (either as a new file or in an existing one, see [Pokémon Brown](configs/pokemon_brown_vars.yaml) for an example)
2. Obtain the ROM hack and place it in the desired path under the [ROM data folder](rom_data). 
3. Go to the [parsers](src/poke_env/emulators/pokemon/parsers.py) and add the required ROM hack. See the `PokemonBrownGameStateParser` as an example. 
4. Go to the [registry](src/poke_env/emulators/pokemon/__init__.py) and add the ROM hack to `VARIANT_TO_GB_NAME`, `_VARIANT_TO_BASE_MAP`, `_VARIANT_TO_PARSER`
5. Run `python dev/create_first_state.py --variant <variant_name>`. This will create a default state. You will not be able to run the `Emulator` on this ROM before doing this. 
6. Run `python dev/dev_play.py --variant <variant_name>` (with the [`gameboy_dev_play_stop` parameter](configs/gameboy_vars.yaml) set to `false`) and proceed through the game until you reach a satisfactory default starting state. Then, open the [config file](configs/gameboy_vars.yaml) and set `gameboy_dev_play_stop` to `true` and save the file. This will trigger a dev mode and ask you for a terminal input. Enter `s default` and you will set that as the new default state. Enter `s initial` as well to save it properly. 

I have provided an [example](https://drive.google.com/file/d/1fsMjkOjpbyeLLNxP3JVaj6uVXycwSAVC/view?usp=sharing) video for this process.

#### Capturing Screens:
The above steps will let you play the game in `debug` mode, but to properly set it up, you need to sync the screen captures by capturing the game's frame at the right moment. This repo uses screen captures and comparison of screen renders to determine state (e.g. menu open, in battle). In Pokémon, the screen markers occur in regular places, and the ROM hacks don't change this much either, making it a reliable way to check for events / flags. 

For the basic regions, run in dev play mode, stop the game at the flag and run `c <region_name>` to save the screen region at that point. The exact screens vary with the base game. The [base classes](src/poke_env/emulators/pokemon/parsers.py) make it clearer what to capture for each named region. 

I have provided an [example](https://drive.google.com/file/d/1EEpoxHAnNwdSMSYcc93xrQCcLzbtVCyX/view?usp=sharing) video for this too. 

If the capture doesn't look right and needs to be shifted, you can use `override_regions`. Follow the example of `battle_enemy_hp_text` for [StarBeasts](src/poke_env/emulators/pokemon/parsers.py). 

You will know that you have filled out all required regions when you can run `python demo.py --variant <variant_name>` without debug mode. 

**Setup Speedrun Guide:**
I've documented the fastest workflow I have found to capturing all the screens for a ROM properly. 

Start by just playing through the game (super high `gameboy_headed_emulation_speed`) and establishing save states for the following:
1. `initial`: Right out of the intro screen with options set to fastest / least animation
2. `starter`: Right before the player needs to make a choice of starter
3. `pokedex`: Not too long after the player obtains the Pokedex, but anywhere you like. 

Then, start with:
```
python dev/dev_play.py --variant <variant> --init_state initial
```
You can tick off the following captures:
* `dialogue_bottom_right`: usually theres something you can interact with in your starting room
* `menu_top_right`: open the start menu
* `pc_top_left`: there is often a PC in your room
* `player_card_middle`: open your player card
* `map_bottom_right`: usually there's a map around you

Then, switch out to the start choice state with `l starter`. Use this state to capture:
* `dialogue_choice_bottom_right`: confirmation message for starter
* `name_entity_top_left`: give the starter a nickname
* `battle_enemy_hp_text`: either a rival battle or just your first pokemon battle
* `battle_player_hp_text`: same
* `pokemon_list_hp_text`: can do once you've got the starter

Then honestly you probably want to exit with `e` and start again at the `pokedex` state with:
```bash
python dev/dev_play.py --variant <variant> --init_state pokedex
```
You'll get a message letting you know what's left. You can finish them all off now. If any of the captures weren't clean and good, you should leave them for the end and override their named screen regions. 

### I want to deepen the state / observation space or give a precise reward
Perhaps you want to engineer the system a little more, like [the initial creators of this framework](https://www.youtube.com/watch?v=DcYLT37ImBY&feature=youtu.be) did. This repo tries to avoid reliance on reading from memory states etc., but certainly supports it at a deep level. See the [memory reader](src/poke_env/emulators/pokemon/parsers.py) state parser to get a sense of how you should go about this. 



## Citation

```bibtex
```
