from pathlib import Path
import subprocess
import sys
import shutil
import os


ROOT_DIR = Path(__file__).resolve().parent
ENV_DIR = ROOT_DIR / ".env"
DATA_ROOT = ROOT_DIR / "data"
ALLZIP_URL = "https://lindat.mff.cuni.cz/repository/server/api/core/items/bec807e1-da5a-4071-befd-9d611bd74c52/allzip?handleId=11372/LRT-2372"


def ensure_virtualenv() -> Path:
    """Create the virtual environment if missing and return python path."""
    if not ENV_DIR.exists():
        subprocess.run([sys.executable, "-m", "venv", str(ENV_DIR)], check=True)
    python_path = ENV_DIR / "bin" / "python"
    alt_python = ENV_DIR / "python"
    if not alt_python.exists():
        alt_python.symlink_to(python_path)
    return python_path


def install_requirements(python_path: Path) -> None:
    """Install dependencies into the virtual environment."""
    requirements = ROOT_DIR / "requirements.txt"
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )


def main() -> None:
    python_path = ensure_virtualenv()
    install_requirements(python_path)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    allzip_path = DATA_ROOT / "allzip.zip"
    muscima_zip_path = DATA_ROOT / "MUSCIMA-pp_v1.0.zip"
    cvc_zip_path = DATA_ROOT / "CVCMUSCIMA_SR.zip"
    root_allzip_path = ROOT_DIR / "allzip.zip"
    root_muscima_zip_path = ROOT_DIR / "MUSCIMA-pp_v1.0.zip"
    root_cvc_zip_path = ROOT_DIR / "CVCMUSCIMA_SR.zip"

    # Download metadata for MUSCIMA++
    subprocess.run(["wget", "-O", str(allzip_path), ALLZIP_URL], check=True)
    subprocess.run(["unzip", str(allzip_path), "-d", str(DATA_ROOT)], check=True)
    if muscima_zip_path.exists():
        subprocess.run(["unzip", str(muscima_zip_path), "-d", str(DATA_ROOT)], check=True)
    muscima_zip_path.unlink(missing_ok=True)
    allzip_path.unlink(missing_ok=True)
    root_allzip_path.unlink(missing_ok=True)
    root_muscima_zip_path.unlink(missing_ok=True)

    # Download the pictures of MUSCIMA++
    from omrdatasettools import Downloader, OmrDataset  # lazy import after install

    downloader = Downloader()
    downloader.download_and_extract_dataset(OmrDataset.CvcMuscima_StaffRemoval, str(DATA_ROOT))
    cvc_ideal_dir = DATA_ROOT / "CvcMuscima-Distortions" / "ideal"
    subprocess.run(["ls", "-la", str(cvc_ideal_dir)], check=True)

    # Extract the data we need and delete the rest
    target_images_dir = DATA_ROOT / "v1.0" / "data" / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)
    for item in cvc_ideal_dir.iterdir():
        dest_path = target_images_dir / item.name
        if dest_path.exists():
            shutil.rmtree(dest_path) if dest_path.is_dir() else dest_path.unlink()
        shutil.move(str(item), str(dest_path))

    shutil.rmtree(DATA_ROOT / "CvcMuscima-Distortions", ignore_errors=True)
    cvc_zip_path.unlink(missing_ok=True)
    root_cvc_zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()