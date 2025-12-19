<div align="center">
  <picture>
    <img alt="Pokémon Environments" src="assets/logo_tilt.png" width="350px" style="max-width: 100%;">
  </picture>
  <br>
  
  **Building Intelligent and General Pokémon Agents**
  
  <br>
    <a href="https://github.com/DhananjayAshok/PokeWorlds/blob/main/LICENSE" target="_blank" rel="noopener noreferrer"><img alt="GitHub" src="https://img.shields.io/badge/license-MIT-blue"></a>
    <a href="https://dhananjayashok.github.io/" target="_blank" rel="noopener noreferrer"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://dhananjayashok.github.io/PokeWorlds/" target="_blank" rel="noopener noreferrer"><img alt="GitHub" src="https://img.shields.io/badge/documentation-pdoc-red"></a>
</div>


<img src="assets/logo.png" width="70px"> is an AI research framework for training and evaluating generally capable agents in the world of Pokémon, complete with flexible Python simulators and unified environment wrappers around Gen I and Gen II Pokémon games. 

![](assets/worlds_random.gif)


Challenge your agents to explore, build general skills and master one of the most iconic game universes ever created.

# Core Features

**Lightweight Environment Parsing:**
We provides simple mechanisms to determine the basic state of the agent and identify specific event triggers that occur in the game, allowing one to form descriptive state spaces and track a broad range of metrics over playthroughs. 

**Abstracted Action Space and Low Level Controllers:**
While all Pokémon games can be played with joystick inputs and a few buttons, not all inputs are meaningful at all times (e.g. when in dialogue, the agent cannot perform any action until the conversation is complete, temporarily reducing the meaningful action space to a single button.)


Another major hurdle to progress is the difficulty of learning how abstract actions (e.g. "Use the Flamethrower attack") correspond to low level game console inputs (e.g. Click 'B' until you are in the 'Battle' Menu, navigate to the 'Fight' button in the menu with the arrow keys and click 'A', then navigate to 'Flamethrower' with the arrow keys and finally click 'A'.)

<img src="assets/logo.png" width="70"> allows language-capable agents to play the game without any awareness of the buttons, and perform actions purely by verbalising its intent (e.g. "Open the bag"). Our low-level controllers then process the request and convert it into the required sequence of button inputs, providing a layer of abstraction. 


**General and "Unleaked" Test Environments:**
<img src="assets/logo.png" width="70"> not only supports classic titles like PokémonRed and PokémonCrystal, but also includes multiple fan-made variants such as [PokémonPrism](https://rainbowdevs.com/title/prism/), that place the agent in completely new regions, sometimes with  fan-made Pokémon ([Fakémon](https://en.wikipedia.org/wiki/Fakemon)). The research value of these fan-made games is considerable:

* Fan-made games are an ideal environment to test the *generalization* capabilities of an agent trained in the original games. To perform well in these unseen environments, agents must learn transferrable skills like battle competence and maze navigation, as opposed to memorizing the solutions to the original games.
* Unlike the original Pokémon games, fan-made games are scarcely documented and so models trained on internet-scale corpora (e.g. Large Language Models) are unlikely to have already acquired a rich base of knowledge regarding the game areas or particular Fakémon. While good performance in PokémonRed may be a result of the backbone model's data being contaminated with walkthroughs and internet guides, the same concern is far less valid for more obscure fan-made games.

# Table of Contents

- [Installation](#Installation)
- [Quickstart](#Quickstart)
- [Developer Tools](#Development)
  - [Custom Starting States](#I-want-to-create-my-own-starting-states)
  - [Descriptive State and Event Tracking](#I-want-to-track-fine-grained-details)
  - [Reward Engineering](#I-want-to-engineer-my-own-reward-functions)
  - [Adding New ROMs](#I-want-to-add-a-new-ROM-Hack)

- [Our Paper](#Check-Out-Our-Paper)

# Installation

The installation consists of four steps:
1. Environment Setup
2. ROM Setup
3. (Optional) Storage Directory Configuration
4. Final test

## Environment Setup
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

Then, clone the <img src="assets/logo.png" width="70"> repo and install it as a `pip` package:
```
git clone https://github.com/DhananjayAshok/PokeWorlds
cd PokeWorlds
uv pip install -e .
```

You can now `import poke_worlds` from anywhere.
## ROM Setup

Next, you must legally acquire ROMs for Pokémon from Nintendo (perhaps by dumping the ROM file from your own catridge). Despite how easy they are to obtain, we discourage any attempts to use <img src="assets/logo.png" width="70"> with unofficialy downloaded ROMs. The following base game ROMs are supported:
* Pokémon Red (save as `PokemonRed.gb`)
* Pokémon Crystal (save as `PokemonCrystal.gbc`)

Additionally, our testing environment uses several Pokémon ROM patches / hacks that alter the game in some way. The official way to acquire these can be obtained is by applying a "patch" to the original ROM. After patching the original ROM, you will be left with a `.gb` or `.gbc` file. Once again, despite their widespread availability, we do not advise you to download the pre-patched ROMs. We support:
* [Pokémon Brown](https://rainbowdevs.com/title/brown/) (save as `PokemonBrown.gbc`)
* [Pokémon Prism](https://rainbowdevs.com/title/prism/) (save as `PokemonPrism.gbc`)
* [Pokémon Fool's Gold](https://www.pokecommunity.com/threads/pok%C3%A9mon-fools-gold-a-hack-of-crystal-where-everything-is-familiar-yet-different.433723/) (save as `PokemonFoolsGold.gbc`)
* [Pokémon Star Beasts](https://www.pokecommunity.com/threads/star-beasts-asteroid-version.530552/) (save as `PokemonStarBeasts.gb`)

Once you have a ROM (`.gb` or `.gbc` file), place it in the appropriate path. For example, the ROM for Pokémon Red should be placed in `rom_data/pokemon_red/PokemonRed.gb`. See the [config folder](configs/) for the expected path to each supported game. 



## Storage Directory Configuration
By default, this project assumes that you can store emulator outputs (logs, screenshots, video captures etc.) in the `storage` folder under the root directory of the project. If you wish to set a different location for storage, edit the appropriate configuration setting in the [config file](configs/private_vars.yaml).

## Test

Check that setup is fine by running (requires a screen to render):
```bash
python demo.py
```
This should open up a GameBoy window where you can play the Pokemon Red game. 

To try a headless test / see how a random agent does, try:
```bash
python demo.py --play_mode random --save_video True
```
The video gets saved to the `sessions` folder of your `storage_dir` directory.


# Quickstart



# Development
This section goes into details on how you would implement new features in <img src="assets/logo.png" width="70">. 

### I want to create my own starting states
Easy. The only question is whether you want to save an mGBA state (perhaps you use cheats to lazily put the agent in a very specific state) or save a PyBoy state directly (i.e. you start from an existing state and play to the new state).

**From mGBA state:**

First, start with mGBA and **make sure** to match the player name and text box frame options from the existing default states. This is vital to ensure the state parsing system works. Play till the point you want to replicate with a state and save the game (go to the start menu and save) in the state you want to restore from. This will make a `variant_ROMNAME.sav` file in the same directory as the rom (which should be in rom_data/variant/). Then run:

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

### I want to track fine-grained details
Maybe you want to enhance the observation space of the agent with information about the current playthrough (e.g. current map ID, enemy team level). Perhaps you want to train text-only / weak visual agents, and parse as much of the screen image as possible into numerical signals / text (e.g. your team stats, bag contents). Some might not even care about their agents, but want to have a sophisticated set of metrics that they can look at to assess goal conditions, judge the quality of a playthrough, or [craft a good reward function](#i-want-to-engineer-my-own-reward-functions). 

Whatever you motivation, <img src="assets/logo.png" width="70"> provides a powerful set of approaches for reading game states, and then allows you to aggregate over these values over time to compute useful metrics for reward assignment and evaluation. 

The first thing to do is detect an event at a moment in time. This is done in subclasses of the `StateParser` [object](src/poke_worlds//emulation/emulator.py) in one of two ways: 

1. **Emulator Screen Captures:** Often particular game states can be cleanly identified by a unique text popup, or some other characteristic marker on the screen. Any of these can be easily captured and checked with the existing parsing system. For example, the current implementation for Pokémon Red has screen captures set up to identify which starter the player chooses. See the [section below](#capturing-screens) for examples of this being done.
2. **Memory Slot Hooks:** A strong alternative is to just directly read statistics from the game's WRAM. Visually inaccessible information (e.g. the attack stats of all Pokémon on the opponents team) are often easy to obtain this way. The only catch is, this method relies on knowing which memory slots to look for. That's easy enough for the original games which have excellent [decompilation guides](https://github.com/pret/pokered/blob/symbols/pokered.sym), but is much harder to do for ROM hacks, which may mess around with the slots arbitrarily. See the [memory reader](src/poke_worlds/emulation/pokemon/parsers.py) state parser to get a sense of how you should go about this. 

These approaches allow your state parsers to give instant-wise decisions or indications when an event has occured. You can then configure your `StateTracker` to use the parser to check for this flag, and store appropriate metrics. See the existing [parsers](src/poke_worlds/emulation/pokemon/parsers.py) and [trackers](src/poke_worlds/emulation/pokemon/trackers.py) for examples. 

### I want to add a new ROM Hack
Setting up a new ROM Hack is an easy process that doesn't take more than an hour once you've understood how. Please do reach out to me if you have any questions, and we can work to merge the new ROM into <img src="assets/logo.png" width="70"> together. 

#### Initial Steps:

0. Set the repo to `debug` mode by editing the [config file](configs/project_vars.yaml)
1. Create a `$variant_rom_data_path` parameter in the [configs](configs) (either as a new file or in an existing one, see [Pokémon Brown](configs/pokemon_brown_vars.yaml) for an example)
2. Obtain the ROM hack and place it in the desired path under the [ROM data folder](rom_data). 
3. Go to the [parsers](src/poke_worlds/emulation/pokemon/parsers.py) and add the required ROM hack. See the `PokemonBrownGameStateParser` as an example. 
4. Go to the [registry](src/poke_worlds/emulation/pokemon/__init__.py) and add the ROM hack to `VARIANT_TO_GB_NAME`, `_VARIANT_TO_BASE_MAP`, `_VARIANT_TO_PARSER`
5. Run `python dev/create_first_state.py --variant <variant_name>`. This will create a default state. You will not be able to run the `Emulator` on this ROM before doing this. 
6. Run `python dev/dev_play.py --variant <variant_name>` (with the [`gameboy_dev_play_stop` parameter](configs/gameboy_vars.yaml) set to `false`) and proceed through the game until you reach a satisfactory default starting state. Then, open the [config file](configs/gameboy_vars.yaml) and set `gameboy_dev_play_stop` to `true` and save the file. This will trigger a dev mode and ask you for a terminal input. Enter `s default` and you will set that as the new default state. Enter `s initial` as well to save it properly. 

I have provided an [example](https://drive.google.com/file/d/1fsMjkOjpbyeLLNxP3JVaj6uVXycwSAVC/view?usp=sharing) video for this process.

#### Capturing Screens:
The above steps will let you play the game in `debug` mode, but to properly set it up, you need to sync the screen captures by capturing the game's frame at the right moment. <img src="assets/logo.png" width="70"> uses screen captures and comparison of screen renders to determine state (e.g. menu open, in battle). In Pokémon, the screen markers occur in regular places, and the ROM hacks don't change this much either, making it a reliable way to check for events / flags. 

For the basic regions, run in dev play mode, stop the game at the flag and run `c <region_name>` to save the screen region at that point. The exact screens vary with the base game. The [base classes](src/poke_worlds/emulation/pokemon/parsers.py) make it clearer what to capture for each named region. 

I have provided an [example](https://drive.google.com/file/d/1EEpoxHAnNwdSMSYcc93xrQCcLzbtVCyX/view?usp=sharing) video for this too. 

If the capture doesn't look right and needs to be shifted, you can use `override_regions`. Follow the example of `battle_enemy_hp_text` for [StarBeasts](src/poke_worlds/emulation/pokemon/parsers.py). 

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

Using this process I'm able to set up all but one capture in [under 10 minutes](https://drive.google.com/file/d/1KkZZe3ON-0EWiBs_EhrAHc9D7lsQmCxW/view?usp=sharing) (the video cuts off with only `pokedex_info_height_text` unassigned because it needs to be manually repositioned as an override region). 

### I want to engineer my own reward functions

<img src="assets/logo.png" width="70"> avoids most domain-knowledge specific reward design, with a motivation of having the agent discover the best policy with minimal guidance. But it's absolutely possible to use your knowledge of the game to create sophisticated reward systems, like [the initial creators of this framework](https://www.youtube.com/watch?v=DcYLT37ImBY&feature=youtu.be) did. 

You'll likely want to gather as much state and trajectory information as possible, for which you should see the [section above](#I-want-to-create-my-own-starting-states).

{NOTE ON WHERE THE REWARD GOES}



# Citation

```bibtex
```
