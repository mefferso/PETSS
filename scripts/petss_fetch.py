import re
import io
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests


NOMADS_PETSS_PROD = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/petss/prod/"


@dataclass
class PetssRunRef:
    date_dir: str          # e.g. "petss.20260213/"
    cycle: str             # e.g. "00"
    csv_tar_url: str       # full URL to petss.t00z.csv.tar.gz


def _list_dir(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def find_latest_petss_csv_tar() -> PetssRunRef:
    """
    Find latest petss.YYYYMMDD directory and the latest cycle tarball inside it.
    Strategy:
      1) list /prod/ and pick max date petss.YYYYMMDD/
      2) list that dir, find petss.t??z.csv.tar.gz entries; choose latest by cycle order 00>06>12>18 if multiple.
    """
    html = _list_dir(NOMADS_PETSS_PROD)
    dirs = re.findall(r'href="(petss\.(\d{8})/)"', html)
    if not dirs:
        raise RuntimeError("No petss.YYYYMMDD/ directories found on NOMADS.")

    # pick latest date
    latest_dir, latest_date = sorted(dirs, key=lambda x: x[1])[-1]
    day_url = NOMADS_PETSS_PROD + latest_dir
    day_html = _list_dir(day_url)

    # find cycle tarballs
    tars = re.findall(r'href="(petss\.t(\d{2})z\.csv\.tar\.gz)"', day_html)
    if not tars:
        raise RuntimeError(f"No petss.t??z.csv.tar.gz found in {day_url}")

    # prefer latest cycle (18,12,06,00) based on availability time; but "latest" is not always 18.
    # We'll choose max by numeric cycle.
    tar_name, cycle = sorted(tars, key=lambda x: int(x[1]))[-1]
    tar_url = day_url + tar_name

    return PetssRunRef(date_dir=latest_dir, cycle=cycle, csv_tar_url=tar_url)


def download_csv_tarball(runref: PetssRunRef, timeout: int = 60) -> bytes:
    r = requests.get(runref.csv_tar_url, timeout=timeout)
    r.raise_for_status()
    return r.content


def extract_csvs_from_tarball(tar_bytes: bytes) -> Dict[str, bytes]:
    """
    Returns dict of {filename: file_bytes} for each CSV member of tar.gz
    """
    out: Dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().endswith(".csv"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            out[member.name] = f.read()
    if not out:
        raise RuntimeError("Tarball contained no CSV files.")
    return out
