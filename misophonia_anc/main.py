import typer

from .model import MisophoniaANCNet

# from ._train_eval_utils import ... # TODO: Import actual functions needed for training and evaluation

app = typer.Typer(help="Misophonia ANC model training and evaluation CLI.")


@app.command()
def train(
    some_param: int = typer.Option(..., help="Some parameter for training"),  # TODO: Add actual parameters
) -> None:
    """
    Train the Misophonia ANC model.
    """  # TODO: Improve docs
    model = MisophoniaANCNet()  # noqa: F841
    raise NotImplementedError("Training function not implemented yet")


@app.command()
def evaluate(
    some_param: int = typer.Option(..., help="Some parameter for evaluation"),  # TODO: Add actual parameters
) -> None:
    raise NotImplementedError("Evaluation function not implemented yet")


if __name__ == "__main__":
    app()
