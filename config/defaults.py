from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = "0"

_C.MODEL.NAME = "ViT-B-16"
_C.MODEL.PRETRAIN_CHOICE = "imagenet"
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False
_C.MODEL.SIE_COE = 1.0

_C.MODEL.METRIC_LOSS_TYPE = "triplet"
_C.MODEL.IF_LABELSMOOTH = "on"
_C.MODEL.IF_WITH_CENTER = "no"
_C.MODEL.DIST_TRAIN = False
_C.MODEL.NO_MARGIN = False

_C.MODEL.ID_LOSS_WEIGHT = 0.25
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.SUPCON_LOSS_WEIGHT = 0.5

# PromptSG related
_C.MODEL.PROMPTSG = CN()
_C.MODEL.PROMPTSG.CMT_DEPTH = 2
_C.MODEL.PROMPTSG.INVERSION_LAYERS = 2
_C.MODEL.PROMPTSG.INVERSION_DROPOUT = 0.1
_C.MODEL.PROMPTSG.TRAIN_MODE = "composed"
_C.MODEL.PROMPTSG.TEST_MODE = "simplified"
_C.MODEL.PROMPTSG.COMPOSED_TEMPLATE = "A photo of a X person"
_C.MODEL.PROMPTSG.SIMPLE_TEMPLATE = "A photo of a person"
_C.MODEL.PROMPTSG.FREEZE_TEXT_ENCODER = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.SIZE_TEST = [256, 128]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PADDING = 10
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAMES = ("market1501",)
_C.DATASETS.ROOT_DIR = ("../data",)

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = "softmax_triplet"
_C.DATALOADER.NUM_INSTANCE = 4

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.SEED = 1234

_C.SOLVER.IMS_PER_BATCH = 64

# learning rates
_C.SOLVER.VISUAL_BASE_LR = 0.000005
_C.SOLVER.NEW_MODULE_BASE_LR = 0.00005

# regularization
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001

# triplet
_C.SOLVER.MARGIN = 0.3

# schedule
_C.SOLVER.MAX_EPOCHS = 60
_C.SOLVER.STEPS = (20, 40)
_C.SOLVER.GAMMA = 0.1

# warmup
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_METHOD = "linear"

# logging / eval / save
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 10

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.EVAL = True
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.RE_RANKING = False
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = "before"
_C.TEST.FEAT_NORM = "yes"
_C.TEST.DIST_MAT = "dist_mat.npy"

# -----------------------------------------------------------------------------
# MISC
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = ""