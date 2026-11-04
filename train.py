"""Train the workspace selected by a Hydra configuration.

The launcher can live beside the source package on servers, or in the source
package itself. Use --verify to check configuration and source paths without
loading the demonstration dataset or importing the robot environment.
"""
from pathlib import Path
import importlib.util
import sys

ENTRY_DIR = Path(__file__).resolve().parent


def _source_root(entry_dir):
    """Find a complete flat or nested source package without changing cwd."""
    candidates = (entry_dir, entry_dir / "EfficientFlow")
    for candidate in candidates:
        if ((candidate / "__init__.py").is_file()
                and all((candidate / name).is_dir() for name in ("config", "policy", "workspace"))):
            return candidate
    checked = "\n".join(f"  {candidate}" for candidate in candidates)
    raise SystemExit(
        "Cannot locate the EfficientFlow source package. Checked:\n"
        f"{checked}\n"
        "Expected __init__.py plus config/, policy/ and workspace/ in one directory. "
        "Keep the source package together; an empty __init__.py beside the launcher is not a fix.")


ROOT = _source_root(ENTRY_DIR)
if not (ROOT / "common" / "train_entry.py").is_file():
    raise SystemExit(
        f"Missing {ROOT / 'common' / 'train_entry.py'}. "
        "Upload common/train_entry.py into the same source package before starting.")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "EfficientFlow", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["EfficientFlow"] = package
spec.loader.exec_module(package)

import hydra
from EfficientFlow.common.train_entry import (
    MAX_STEPS, get_ws_x_center, register_resolvers, run_workspace,
)

register_resolvers()
VERIFY_ONLY = "--verify" in sys.argv
if VERIFY_ONLY:
    sys.argv.remove("--verify")


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="EfficientFlow")
def main(cfg):
    run_workspace(cfg, source_root=ROOT, entry_file=__file__, verify_only=VERIFY_ONLY)


if __name__ == "__main__":
    main()
