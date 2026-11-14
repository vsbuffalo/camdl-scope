"""Pack the static-demo frontend build into a release tarball plus its SHA-256.

Input: a Vite build made with ``VITE_DEMO=1`` and ``--base ./`` (``make
demo-bundle`` writes it to ``web/dist-demo``). Output, in ``--out-dir``:

- ``camdl-scope-demo-<tag>.tar.gz`` — the build's files (``index.html``,
  ``assets/``, ``favicon.svg``) at the archive root, so ``tar -xzf <it> -C
  <site-dir>`` puts ``index.html`` directly in ``<site-dir>``. It carries no
  ``snap/``: the consumer adds the snapshot written by
  ``scripts/make_demo_snapshot.py`` beside ``index.html``.
- ``camdl-scope-demo-<tag>.tar.gz.sha256`` — one ``<hex>  <filename>`` line, the
  format ``sha256sum -c`` / ``shasum -a 256 -c`` check.

The archive is deterministic for a given build directory: entries are sorted;
mtimes, owners and modes are normalised; and the gzip header carries no
timestamp or filename. Two packs of the same build are byte-identical, so the
published hash identifies the frontend files alone.

Usage::

    python3 scripts/pack_demo_bundle.py --dist web/dist-demo --tag v0.1.0 --out-dir dist
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import tarfile
from pathlib import Path


def _normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def pack(dist: Path, tag: str, out_dir: Path) -> tuple[Path, str]:
    if not (dist / "index.html").is_file():
        raise SystemExit(f"{dist}/index.html not found: build the demo frontend first")
    if (dist / "snap").exists():
        raise SystemExit(
            f"{dist}/snap exists: the release bundle must not carry a snapshot "
            "(it would publish whatever fit store it was made from)"
        )
    name = f"camdl-scope-demo-{tag}"
    files = sorted(dist.rglob("*"))

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in files:
            arc = p.relative_to(dist).as_posix()
            info = _normalise(tar.gettarinfo(str(p), arcname=arc))
            if p.is_file():
                with p.open("rb") as fh:
                    tar.addfile(info, fh)
            else:
                tar.addfile(info)

    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / f"{name}.tar.gz"
    with tarball.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(buf.getvalue())
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    (out_dir / f"{tarball.name}.sha256").write_text(f"{digest}  {tarball.name}\n")
    return tarball, digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", type=Path, required=True, help="the VITE_DEMO=1 build directory")
    ap.add_argument("--tag", required=True, help="release tag, e.g. v0.1.0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    tarball, digest = pack(args.dist, args.tag, args.out_dir)
    print(f"{digest}  {tarball}")


if __name__ == "__main__":
    main()
