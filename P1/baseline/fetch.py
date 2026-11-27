"""Fetch pinned validation data and stage model refs for the unmodified upstream scorer."""
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from pilot_io import DATA, DATA_REV, digest
from score import WEIGHTS


def fetch():
    endpoint = os.environ.get("HF_ENDPOINT","https://huggingface.co").rstrip("/")
    DATA.mkdir(exist_ok=True)
    assets = {"metadata-validation.parquet":"c022eff66ce7241c8f5eaf8a1314b9fdbf36a562655b1a69a3c8668452c6d3db",
              "images/validation/validation_1.zip":"8eb57656d6c424b9451240d5f401462107889f2c4781fb3e5f827f7e73a197ea",
              "images/validation/validation_2.zip":"5349f894b1b1571fbe1aed6a3288bdf9f8e4dbf2d8a6fbe00274477f276da48c",
              "ImageRewardDB.py":None}
    for entry, checksum in assets.items():
        dest = DATA/Path(entry).name
        if dest.exists() and (checksum is None or digest(dest) == checksum):
            continue
        part = dest.with_suffix(dest.suffix+".part")
        url = f"{endpoint}/datasets/zai-org/ImageRewardDB/resolve/{DATA_REV}/{entry}?download=true"
        subprocess.run(["curl","-fSL","--retry","3",url,"-o",str(part)],check=True)
        if checksum and digest(part)!=checksum:
            raise ValueError(f"Invalid checksum for {entry}")
        part.replace(dest)
    for repo,revision,name,checksum in WEIGHTS:
        path = Path(hf_hub_download(repo,name,revision=revision))
        if digest(path)!=checksum:
            raise ValueError(f"Invalid published weight: {name}")
        # Upstream requests 'main'. Give only this project's offline cache a pinned main.
        own = DATA/"model-cache"/("models--"+repo.replace("/","--"))
        snapshot = own/"snapshots"/revision
        snapshot.mkdir(parents=True,exist_ok=True)
        target = snapshot/name
        if not target.exists():
            target.symlink_to(path.resolve())
        assert digest(target)==checksum
        (own/"refs").mkdir(exist_ok=True)
        (own/"refs"/"main").write_text(revision)


if __name__ == "__main__":
    fetch()

