DATA_FOLDER = "../data/stl10/stl10_binary/"
DATA_FILE   = "unlabeled_X.bin"

MODEL_FOLDER = "../models/"

TRAINING_SET_SIZE      = 80_000 # 80,000 images for training -> 20,000 images for validation
VALIDATION_SET_SIZE    = 20_000
BATCH_SIZE_PER_REPLICA = 16 # batch size per gpu replica (e.g., 64 for 1 gpu, 128 for 2 gpus, etc.)
EPOCHS                 = 50