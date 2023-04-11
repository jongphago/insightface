from easydict import EasyDict as edict
from utils.utils_config import return_pairs_path

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)  #
config.network = "r50"  #
config.resume = False
config.output = None
config.embedding_size = 512  #
config.sample_rate = 1.0  #
config.fp16 = True
config.momentum = 0.9  #
config.weight_decay = 5e-4  #
config.batch_size = 128
config.lr = 0.02  #
# FIXME
config.verbose = 360
config.dali = False
config.save_all_states = True
config.pretrained = True
config.dropout = 0.0  #

# FIXME
config.rec = "/home/jupyter/data/face-image/train_aihub_family"
# config.rec = "/home/jupyter/data/face-image/valid_aihub_family"
config.num_classes = 2154
config.num_image = 93006
config.num_epoch = 5  #
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.task = "family"

config.path = edict()
path = config.path
path.work_dirs = "/home/jongphago/insightface/work_dirs"
path.checkpoint = ""

config.data = edict()
config.data.image_size = 112
config.data.aihub_mean = [0.5444, 0.4335, 0.3800]
config.data.aihub_std = [0.2672, 0.2295, 0.2156]
config.data.pairs_path = return_pairs_path  # function

config.data.test = edict()
config.data.test.path = edict()

test = config.data.test
test.split = "test"
test.task = "family"
test.batch_size = 200
test.num_workers = 4
test.path.aihub_dataroot = "/home/jupyter/insightface/data/face-image/test_aihub_family"
config.data.test.path.pairs_path = config.data.pairs_path
