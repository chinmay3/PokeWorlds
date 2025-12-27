echo "Testing emulator..."
python demos/emulator.py --play_mode random
echo "Testing game variants ..."
python demos/emulator.py --play_mode random --variant pokemon_brown
python demos/emulator.py --play_mode random --variant pokemon_crystal
echo "Testing environment..."
python demos/environment.py --play_mode random