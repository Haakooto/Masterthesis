"""
File used to delete model folders.
Be carful with this file, basically rewrite it everytime you want to use it.
"""

import train_models as tm
from layers.local_learning import LocalLearning
import os
import shutil

good = 0
bad = 0
killed = 0

for model_name in tm.existing_models():
    # print(model_name)
    if not os.path.isfile(f"{tm.hebb_path(model_name)}/ds_evolve.png"): 
        shutil.rmtree(f"{tm.hebb_path(model_name)}")
        killed += 1
        # continue
    # model = LocalLearning(path=tm.hebb_path(model_name))
    # model.load_model(name=model_name, verbose=False)
    # # print(model)
    # model.unconverged_neurons()
    # model.save_model(verbose=False)
    # # print(model)

    # print(f"Model {model_name} is good: {model.good_model}")

    # if not model.good_model:
    #     bad += 1
    # else:
    #     good += 1

    # if not model.good_model: model.purge(complete=True)

print(f"Good: {good}, Bad: {bad}, Killed: {killed}")
# print(f"Good: {good}, bad: {bad}")