import click

from .commands.analyze import _analyze
from .commands.export import _export
from .commands.ocr import _ocr
from .commands.prune import _prune
from .commands.segment import _segment
from .commands.evaluate import _evaluate
from .commands.rank import _rank

cli = click.CommandCollection(sources=[
    _analyze,
    _export,
    _ocr,
    _prune,
    _segment,
    _evaluate,
    _rank,
])

if __name__ == "__main__":
    cli()