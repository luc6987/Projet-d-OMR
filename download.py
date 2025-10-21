from omrdatasettools import Downloader, OmrDataset
import shutil
from pathlib import Path
import os
# Download metadata for MUSCIMA++
os.system("wget -O allzip.zip https://lindat.mff.cuni.cz/repository/server/api/core/items/bec807e1-da5a-4071-befd-9d611bd74c52/allzip?handleId=11372/LRT-2372")
os.system("unzip allzip.zip")
os.system("unzip MUSCIMA-pp_v1.0.zip")
os.system("rm MUSCIMA-pp_v1.0.zip")
os.system("rm allzip.zip")

# Download the pictures of MUscima++
downloader = Downloader()
downloader.download_and_extract_dataset(OmrDataset.CvcMuscima_StaffRemoval, "data")
os.system("ls -la data/CvcMuscima-Distortions/ideal")

# extract the data we need and delete the rest
Path('v1.0/data/images').mkdir(parents=True, exist_ok=True)
for item in Path('data/CvcMuscima-Distortions/ideal').iterdir():
    dest_path = Path('v1.0/data/images') / item.name
    if dest_path.exists():
        shutil.rmtree(dest_path) if dest_path.is_dir() else dest_path.unlink()
    shutil.move(str(item), str(dest_path))
shutil.rmtree(Path('data/CvcMuscima-Distortions'))
os.system("rm -rf data")
os.system("rm CVCMUSCIMA_SR.zip")

