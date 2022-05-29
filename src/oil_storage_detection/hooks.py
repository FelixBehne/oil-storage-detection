import logging
import warnings

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node


def say_hello(node: Node):
    """An extra behaviour for a node to say hello before running."""
    print(f"Hello from {node.name}")


class TransformerHooks:
    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @hook_impl
    def before_node_run(self, node: Node):
        # adding extra behaviour to a single node

        say_hello(node)

    @hook_impl
    def before_pipeline_run(self, catalog: DataCatalog) -> None:

        # Added because the numpy and sklearn versions clash for np.bool type of checks
        warnings.simplefilter("ignore", category=DeprecationWarning)
