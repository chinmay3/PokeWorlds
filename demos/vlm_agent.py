from poke_worlds import AVAILABLE_GAMES, get_environment
import click
from poke_worlds.execution.supervisor import SimpleSupervisor
from poke_worlds.execution.pokemon.executors import PokemonExecutor
        
        
@click.command()
@click.option("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", type=str)
@click.option("--init_state", default="starter", type=str)
@click.option("--game_variant", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--mission", default="Professor oak has invited you into his lab and offered you a choice of starter pokemon from the bench. You are to select a starter, leave the building from the bottom, and keep playing until you get the first gym badge.", type=str)
@click.option("--visual_context", default=None, type=str)
@click.option("--max_steps", default=1000, type=int)
def do(model_name, init_state, game_variant, mission, visual_context, max_steps):
    short_model = model_name.split("/")[-1]
    environment = get_environment(game=game_variant, environment_variant="default", controller_variant="state_wise", 
                                        save_video=True, max_steps=max_steps,
                                            init_state=init_state, session_name=f"vlm_demo_{short_model}", headless=True)
    vl = SimpleSupervisor(game=game_variant, environment=environment, executor_class=PokemonExecutor, model_name=model_name)
    vl.play(mission=mission, visual_context=visual_context)
    environment.close()

if __name__ == "__main__":
    do()
