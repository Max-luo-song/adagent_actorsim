
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

from typing import Dict, List
pass  # Path is provided by the portable boundary above
import logging
import pickle
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_merge_submission_pickles"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    tokens = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs_path,
        sensor_config=SensorConfig.build_no_sensors(),
    ).tokens

    # submission_pickles: List[str] = instantiate(cfg.submission_pickles)

    merged_list: List[Dict[str, Trajectory]] = []
    for submission_pickle in cfg.submission_pickles:
        with open(submission_pickle, "rb") as f:
            output_list: List[Dict[str, Trajectory]] = pickle.load(f)["predictions"]
        for output in output_list:
            assert set(output.keys()) == set(tokens), f"Submission pickle {submission_pickle} has invalid scene tokens!"
        merged_list.extend(output_list)
    logger.info(f"Merged {len(merged_list)} submissions from {len(cfg.submission_pickles)} pickles")

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "predictions": merged_list,
    }

    # pickle and save dict
    filename = os.path.join(save_path, "submission.pkl")
    with open(filename, "wb") as file:
        pickle.dump(submission, file)
    logger.info(f"Your submission filed was saved to {filename}")


if __name__ == "__main__":
    main()
