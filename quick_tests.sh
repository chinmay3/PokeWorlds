variants=("pokemon_red" "pokemon_crystal" "pokemon_brown" "pokemon_prism" "pokemon_fools_gold" "pokemon_starbeasts")
echo "Testing emulator..."
for variant in "${variants[@]}"; do
    echo "  Variant: $variant"
    python demos/emulator.py --play_mode random --variant $variant
done
echo "Testing environment..."
python demos/environment.py --play_mode random