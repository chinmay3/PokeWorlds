from poke_worlds import (
    AVAILABLE_GAMES,
    get_environment,
    get_benchmark_tasks,
    get_test_environment,
)
import click
from poke_worlds.utils import load_parameters
from poke_worlds.execution.supervisor import SimpleSupervisor
from poke_worlds.execution.pokemon.executors import PokemonExecutor
from poke_worlds.execution.pokemon.reports import SimplePokemonExecutionReport
from tqdm import tqdm
import pandas as pd


def run_task(row, max_resets, controller_variant, **emulator_kwargs):
    success = False
    n_resets = 1
    n_steps = 0
    mission = row["task"]
    environment = get_test_environment(
        row=row, controller_variant=controller_variant, **emulator_kwargs
    )
    supervisor = SimpleSupervisor(
        game=row["game"],
        environment=environment,
        executor_class=PokemonExecutor,
        execution_report_class=SimplePokemonExecutionReport,
    )
    supervisor.setup_play(
        mission=mission,
        initial_visual_context="You are seeing a screenshot of the game.",
    )
    while n_resets < max_resets + 1:
        supervisor_report = supervisor.play()
        if len(supervisor_report.execution_reports) > 0:
            last_execution_report = supervisor_report.execution_reports[-1]
            # updated n_steps
            last_state = last_execution_report.get_state_infos()[
                -1
            ]  # TODO: could this be empty?
            step_count = last_state["core"]["n_steps"]
            n_steps += step_count
            # check the last execution report in the supervisor report. It is success only if exit code is 2
            if last_execution_report.exit_code == 2:
                success = True
                break
            else:
                n_resets += 1
        else:
            n_resets += 1
    environment.close()
    return success, n_resets - 1, n_steps


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option(
    "--controller_variant",
    default="state_wise",
    type=str,
)
@click.option("--save_video", type=bool, default=True)
@click.option("--max_resets", default=3, type=int)
@click.option("--max_steps", default=1000, type=int)
def do(game, controller_variant, save_video, max_resets, max_steps):
    project_parameters = load_parameters()
    executor_vlm_name = project_parameters["executor_vlm_model"]
    model_save_name = executor_vlm_name.split("/")[-1].lower()
    session_name = f"benchmark_zero_shot_{model_save_name}"
    headless = True
    emulator_kwargs = {
        "headless": headless,
        "save_video": save_video,
        "session_name": session_name,
        "max_steps": max_steps,
    }
    benchmark_tasks = get_benchmark_tasks(game=game)
    results = []
    columns = ["game", "task", "success", "n_resets", "n_steps"]
    for _, row in tqdm(benchmark_tasks.iterrows(), total=len(benchmark_tasks)):
        success, n_resets, n_steps = run_task(
            row=row,
            max_resets=max_resets,
            controller_variant=controller_variant,
            **emulator_kwargs,
        )
        results.append([row["game"], row["task"], success, n_resets, n_steps])
    df = pd.DataFrame(results, columns=columns)
    save_path = f"benchmark_zero_shot_{game}_{model_save_name}.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved benchmark results to {save_path}")


if __name__ == "__main__":
    do()
