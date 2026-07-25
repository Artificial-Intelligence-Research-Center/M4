"""上傳新資料集 — 壓縮檔安全解壓 + 自動找出資料集根目錄 (供 web UI 使用).

流程: 收檔 → 解壓到 data/<name>/ → 自動偵測真正的 <root>/{train,val,test}
      → dataset_registry.validate() 檢查 MedClaw 是否吃得下。

安全性: 只信任壓縮檔內的相對路徑, 拒收 `..` / 絕對路徑 / symlink / 硬連結
        (zip-slip 防護), 並限制單檔與總解壓大小。
"""
from __future__ import annotations

import os
import re
import shutil
import tarfile
import zipfile

from . import dataset_registry as reg

_SPLITS = ("train", "val", "test")
_MAX_DEPTH = 3                              # 往下找 train/val/test 的層數
_MAX_TOTAL_BYTES = 64 * 1024 ** 3           # 解壓總量上限 (zip bomb 防護)
_ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")


class IngestError(Exception):
    """上傳/解壓過程中可直接顯示給使用者的錯誤。"""


def archive_ext(filename: str) -> str | None:
    low = filename.lower()
    return next((e for e in sorted(_ARCHIVE_EXT, key=len, reverse=True)
                 if low.endswith(e)), None)


def safe_name(name: str) -> str:
    """資料集目錄名: 只留安全字元, 避免路徑穿越。"""
    name = os.path.basename(name.strip()).strip(". ")
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("_")
    return name[:80]


def default_name(filename: str) -> str:
    ext = archive_ext(filename) or ""
    return safe_name(filename[:-len(ext)] if ext else filename) or "dataset"


def _check_member(rel: str, dest: str) -> str:
    """驗證壓縮檔成員的相對路徑, 回傳解壓目標的絕對路徑。"""
    rel = rel.replace("\\", "/")
    if rel.startswith("/") or os.path.isabs(rel) or ".." in rel.split("/"):
        raise IngestError(f"壓縮檔含有不安全的路徑，已中止: {rel}")
    target = os.path.realpath(os.path.join(dest, rel))
    if target != dest and not target.startswith(dest + os.sep):
        raise IngestError(f"壓縮檔含有跳出目標目錄的路徑，已中止: {rel}")
    return target


def extract(archive_path: str, dest: str) -> None:
    """把壓縮檔安全解壓到 dest (dest 必須已存在)。"""
    dest = os.path.realpath(dest)
    total = 0
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                target = _check_member(info.filename, dest)
                if info.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                total += info.file_size
                if total > _MAX_TOTAL_BYTES:
                    raise IngestError("解壓後容量超過上限，已中止")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        return

    try:
        tf = tarfile.open(archive_path)
    except tarfile.TarError as e:
        raise IngestError(f"無法辨識的壓縮格式（支援 {', '.join(_ARCHIVE_EXT)}）：{e}") from e
    with tf:
        for m in tf.getmembers():
            target = _check_member(m.name, dest)
            if m.issym() or m.islnk():
                continue                      # 連結一律略過
            if m.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            if not m.isfile():
                continue
            total += m.size
            if total > _MAX_TOTAL_BYTES:
                raise IngestError("解壓後容量超過上限，已中止")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def locate_root(base: str, max_depth: int = _MAX_DEPTH) -> str | None:
    """在 base 底下找出真正含 train/ 的資料集根目錄 (壓縮檔常多包一層)。"""
    def has_train(d: str) -> bool:
        return os.path.isdir(os.path.join(d, "train"))

    if has_train(base):
        return base
    frontier = [(base, 0)]
    while frontier:
        cur, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        try:
            names = sorted(os.listdir(cur))
        except OSError:
            continue
        for n in names:
            if n.startswith(("__MACOSX", ".")):
                continue
            d = os.path.join(cur, n)
            if not os.path.isdir(d):
                continue
            if has_train(d):
                return d
            frontier.append((d, depth + 1))
    return None


def _lift(root: str, dest: str) -> None:
    """把偵測到的資料集根目錄內容搬到 dest 頂層 (同一檔案系統, 用 rename)。"""
    if os.path.realpath(root) == os.path.realpath(dest):
        return
    for n in os.listdir(root):
        src = os.path.join(root, n)
        dst = os.path.join(dest, n)
        if os.path.exists(dst):
            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
        shutil.move(src, dst)
    # 清掉搬空後留下的中間層目錄
    top = os.path.realpath(dest)
    cur = os.path.realpath(root)
    while cur != top and cur.startswith(top + os.sep):
        parent = os.path.dirname(cur)
        try:
            os.rmdir(cur)
        except OSError:
            break
        cur = parent


def ingest(archive_path: str, data_root: str, name: str,
           overwrite: bool = False) -> dict:
    """解壓 + 整理 + 驗證。回傳 validate() 的報告, 另附 dest / lifted_from。

    失敗 (IngestError) 時會清掉自己建立的目錄, 不留半成品。
    """
    name = safe_name(name)
    if not name:
        raise IngestError("資料集名稱無效")
    data_root = os.path.realpath(data_root)
    dest = os.path.join(data_root, name)
    if os.path.realpath(os.path.dirname(dest)) != data_root:
        raise IngestError("資料集名稱無效")

    existed = os.path.exists(dest)
    if existed and not overwrite:
        raise IngestError(f"data/{name} 已存在。請換個名稱，或勾選「覆蓋同名資料集」。")
    if existed:
        if not os.path.isdir(dest):
            raise IngestError(f"data/{name} 已存在且不是目錄")
        shutil.rmtree(dest)

    os.makedirs(dest, exist_ok=False)
    try:
        extract(archive_path, dest)
        root = locate_root(dest)
        if root is None:
            raise IngestError(
                "壓縮檔裡找不到 train/ 目錄。MedClaw 需要的結構是："
                "<資料集>/{train,val,test}/<類別名>/*.jpg")
        lifted = None if os.path.realpath(root) == os.path.realpath(dest) \
            else os.path.relpath(root, dest)
        _lift(root, dest)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    reg.invalidate(data_root)
    report = reg.validate(dest)
    report["dest"] = dest
    report["rel_path"] = os.path.relpath(dest, data_root)
    report["lifted_from"] = lifted       # 壓縮檔多包的那層 (已攤平), 供訊息提示
    report["overwritten"] = existed
    return report
