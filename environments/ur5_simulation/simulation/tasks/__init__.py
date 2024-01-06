"""Ravens tasks."""

from .align_box_corner import AlignBoxCorner
from .assembling_kits import AssemblingKits
from .assembling_kits import AssemblingKitsEasy
from .assembling_kits_seq import AssemblingKitsSeqSeenColors
from .assembling_kits_seq import AssemblingKitsSeqUnseenColors
from .assembling_kits_seq import AssemblingKitsSeqFull
from .block_insertion import BlockInsertion
from .block_insertion import BlockInsertionEasy
from .block_insertion import BlockInsertionNoFixture
from .block_insertion import BlockInsertionSixDof
from .block_insertion import BlockInsertionTranslation
from .manipulating_rope import ManipulatingRope
from .align_rope import AlignRope
from .packing_boxes import PackingBoxes
from .packing_shapes import PackingShapes
from .packing_boxes_pairs import PackingBoxesPairsSeenColors
from .packing_boxes_pairs import PackingBoxesPairsUnseenColors
from .packing_boxes_pairs import PackingBoxesPairsFull
from .packing_google_objects import PackingSeenGoogleObjectsSeq
from .packing_google_objects import PackingUnseenGoogleObjectsSeq
from .packing_google_objects import PackingSeenGoogleObjectsGroup
from .packing_google_objects import PackingUnseenGoogleObjectsGroup
from .palletizing_boxes import PalletizingBoxes
from .place_red_in_green import PlaceRedInGreen
from .put_block_in_bowl import PutBlockInBowlSeenColors
from .put_block_in_bowl import PutBlockInBowlUnseenColors
from .put_block_in_bowl import PutBlockInBowlFull
from .stack_block_pyramid import StackBlockPyramid
from .stack_block_pyramid_seq import StackBlockPyramidSeqSeenColors
from .stack_block_pyramid_seq import StackBlockPyramidSeqUnseenColors
from .stack_block_pyramid_seq import StackBlockPyramidSeqFull
from .sweeping_piles import SweepingPiles
from .separating_piles import SeparatingPilesSeenColors
from .separating_piles import SeparatingPilesUnseenColors
from .separating_piles import SeparatingPilesFull
from .task import Task
from .towers_of_hanoi import TowersOfHanoi
from .towers_of_hanoi_seq import TowersOfHanoiSeqSeenColors
from .towers_of_hanoi_seq import TowersOfHanoiSeqUnseenColors
from .towers_of_hanoi_seq import TowersOfHanoiSeqFull

names = {
    # demo conditioned
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi,

    # goal conditioned
    'align-rope': AlignRope,
    'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
    'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
    'assembling-kits-seq-full': AssemblingKitsSeqFull,
    'packing-shapes': PackingShapes,
    'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
    'packing-boxes-pairs-full': PackingBoxesPairsFull,
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
    'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
    'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
    'put-block-in-bowl-unseen-colors': PutBlockInBowlUnseenColors,
    'put-block-in-bowl-full': PutBlockInBowlFull,
    'stack-block-pyramid-seq-seen-colors': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-unseen-colors': StackBlockPyramidSeqUnseenColors,
    'stack-block-pyramid-seq-full': StackBlockPyramidSeqFull,
    'separating-piles-seen-colors': SeparatingPilesSeenColors,
    'separating-piles-unseen-colors': SeparatingPilesUnseenColors,
    'separating-piles-full': SeparatingPilesFull,
    'towers-of-hanoi-seq-seen-colors': TowersOfHanoiSeqSeenColors,
    'towers-of-hanoi-seq-unseen-colors': TowersOfHanoiSeqUnseenColors,
    'towers-of-hanoi-seq-full': TowersOfHanoiSeqFull,
}
