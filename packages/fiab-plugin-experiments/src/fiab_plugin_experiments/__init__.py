from fiab_core.fable import BlockFactoryId
from fiab_core.tools.blocks import QubedBlockBuilder
from fiab_core.tools.plugins import QubedPluginBuilder

from fiab_plugin_experiments.blocks import EkdExample, WindPointDirection

blocks: dict[BlockFactoryId, QubedBlockBuilder] = {
    BlockFactoryId("ekdExample"): EkdExample(),
    BlockFactoryId("windPointDirection"): WindPointDirection(),
}

# NOTE the runtime wheel is requested per block (see blocks.runtime_environment), as it is not installed at compile time
plugin = QubedPluginBuilder(block_builders=blocks, base_environment=[]).as_plugin()
