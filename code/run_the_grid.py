"""
File for running the grid search.
As with grid_search.py, dont delete functions from this file.
Is meant to be secure and easy to use.
Multiple threads not implemented, and maybe recomended to never do.
"""

from train_models import main as train_main
import utils.grid_search as grid_search
from utils.argparsing import train_argparser
from utils.telebot import send_msg
import utils.utils as utils

famy_path, hebb_path, clas_path, existing_models = utils.path_utils_pack

from socket import gethostname
import pandas as pd
import threading
import torch, time


def get_next_hebb_model(model_getter):
    """
    Returns the next hebbian model that is not trained yet.
    """
    models, fam = model_getter()
    try:
        living_models = set(pd.read_csv(f"{hebb_path(fam=fam)}family_record.csv")["name"])
    except FileNotFoundError:
        living_models = set()
    living_models = living_models.union(existing_models(fam=fam))
    model_cnt = len(models)
    for idx, name in enumerate(models.keys()):
        if name not in living_models:
            args = train_argparser(models[name].split())
            return args, f"{idx}/{model_cnt}"
    return None, None

def get_next_clas_model(hebbian, model_getter):
    """
    Returns the next classification model that is not trained yet.
    """
    models, fam = model_getter(hebbian)
    try:
        living_models = set(pd.read_csv(f"{clas_path(hebbian, fam=fam)}family_record.csv")["name"])
    except FileNotFoundError:
        living_models = set()
    living_models = living_models.union(existing_models(hebbian, fam=fam))
    model_cnt = len(models)
    for idx, name in enumerate(models.keys()):
        if name not in living_models:
            args = train_argparser(models[name].split())
            return args, f"{idx}/{model_cnt}"
    return None, None

def secure_run(model_getter, wait_time=2):
    """
    Lol it is called 'secure' because it is meant to be secure, but god knows if it is.
    Runs a while loop that checks if the next model is trained, and if not, trains it.
    """
    stop = False
    while not stop:
        model, idx = model_getter()
        if model is None:
            stop = True
            continue
        try:
            print(f"Starting {idx}: {model.name}")
            train_main(model)
        except KeyboardInterrupt:
            print("Stops training the model.\nRe-interupt before next model starts to completely stop the program.")
            stop = True
        except FileExistsError:
            print("File exists error. Skipping.")
        except Exception as e:
            send_msg(f"Something went wrong with grid-search on {gethostname()} at model {idx}.")
            send_msg(f"Error: {e}")
            stop = True
            raise e
        finally:
            print("Next!\n\n")
            time.sleep(wait_time)

def hebb_run(GS_func):
    """Hebbians are easy to run, just need to give it a model getter.
    This also works with BPs, as they only need a model getter."""
    secure_run(lambda: get_next_hebb_model(GS_func))

def clas_run(hebb_func, GS_func):
    """Classifications are a bit more complicated, need to give it a hebbian getter and a model getter."""
    hebbians = hebb_func()[0].keys()
    for idx, hebbian in enumerate(hebbians):
        if hebbian in existing_models():
            print(f"Starting Hebbian {idx}/{len(hebbians)}: {hebbian}")
            secure_run(lambda: get_next_clas_model(hebbian, GS_func))

def restricted_clas_run(hebb_func, GS_func, restrictor):
    """
    Same as clas_run, but with a restriction on the hebbian models, like if you only want models with a certain K or ratio.
    """
    hebbians = hebb_func()[0].keys()
    for idx, hebbian in enumerate(hebbians):
        if hebbian in existing_models():
            if not restrictor(read_params(hebbian)):
                print(f"Starting Hebbian {idx}/{len(hebbians)}: {hebbian}")
                secure_run(lambda: get_next_clas_model(hebbian, GS_func))

def read_params(hebbian):
    """Reads the 'readable_parameters.txt' file and returns a dictionary with the parameters."""
    with open(f"{hebb_path()}{hebbian}/readable_parameters.txt") as f:
        params = f.read().strip().split("\n")[3:]
        params = {p.split(":")[0].strip(): p.split(":", 1)[1].strip() for p in params}
    return params

def PM_restrictor(params):
    """Restrictor for ProjectMercury. Returns True if the hebbian is resticted, False if it should be trained."""
    if float(params["ratio"]) >= 0.5:
        if int(params["width"]) == 4:
            if int(params["p"]) > 2:
                if float(params["ratio"]) < 0.9:
                    return False
    return True

def Activate_restrictor(params):
    """Restrictor for Activation. Returns True if the hebbian is resticted, False if it should be trained."""
    if float(params["ratio"]) >= 0.8:
        return False
    return True


if __name__ == "__main__":
    # hebb_run(grid_search.ProjectMercuryBP)
    # clas_run(grid_search.ProjectMercuryHebbian, grid_search.ProjectMercuryClassifiers)
    # restricted_clas_run(grid_search.ProjectMercuryHebbian, grid_search.ProjectMercuryClassifiers_restricted, PM_restrictor)

    # hebb_run(grid_search.Activation_Hebb)
    # hebb_run(grid_search.Activation_BP)
    # clas_run(grid_search.Activation_Hebb, grid_search.Activation_Classifiers)
    restricted_clas_run(grid_search.Activation_Hebb, grid_search.Activation_Classifiers, Activate_restrictor)

    send_msg(f"Done with grid search on {gethostname()}.")

