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
python tests/demo.py
```
This should open up a GameBoy window where you can play the Pokemon Red game. 


## Quickstart


## Core Features


## Development
You may want to create your own initial states to start the emulation from. 

First, start with mGBA and save the game in the state you want to restore from. This will make a ROMNAME.sav file. Then run:

```bash
python tests/gameboy/save_state.py
```

This will save the state to `tmp.state`

Alternatively, run the pyboy simulator with:
```bash 
python tests/gameboy/dev.py
```

This will run the game with the option to enter dev mode. If you want to save a particular state, go to [the gameboy configs](configs/gameboy_vars.yaml) *while* playing the game (at the state you want to save), change the `gameboy_dev_play_stop` parameter to `true` and then check the terminal. You will get a message with the possible dev actions. 

* `s`: saves the state (useful for creating fixed starting points)
* `c`: captures the screen in a particular [named region]() LINK TO DOCS at the current game frame (useful for creating reference images of particular states e.g. menu/pc open, dialogues that trigger on particular milestones or events etc.) WRITE A README WITH EXAMPLES ON BOTH OF THESE





## Citation

```bibtex
```
