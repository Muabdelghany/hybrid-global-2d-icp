#!/usr/bin/env python3
"""Structural self-check for the reproducibility package.

Run from anywhere:  python3 verify_package.py
Exits non-zero if any check fails, so it can gate a release.

It checks what a reader would hit first: broken links, build artifacts, a checksum
manifest that does not verify, generators whose data is missing, paths quoted in the
documentation that do not exist, and machine-local or confidential strings that should
not have shipped. It does not re-run the solver; for that, compare a solve against
model/VERIFICATION_CASE.json.
"""
import collections, hashlib, os, re, shutil, subprocess, sys

def walk(root):
    """os.walk with .git pruned. Git's object store is not payload: it differs between
    any two clones of the same tree and would otherwise produce phantom findings."""
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".git"]
        yield d, dirs, fs

ROOT = os.path.dirname(os.path.abspath(__file__))
fail = []

# `--clean` first removes the build junk that macOS and Python regenerate on their own
# (.DS_Store from Finder, __pycache__ from any import), then verifies. Without it those
# files are reported rather than deleted, so a check run never mutates the package.
if "--clean" in sys.argv:
    import shutil
    removed = 0
    for _d, _dirs, _fs in walk(ROOT):
        for _x in list(_dirs):
            if _x == "__pycache__":
                shutil.rmtree(os.path.join(_d, _x), ignore_errors=True)
                _dirs.remove(_x); removed += 1
        for _f in _fs:
            if _f == ".DS_Store" or _f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(_d, _f)); removed += 1
                except OSError:
                    pass
    print("cleaned %d build-junk item(s)\n" % removed)


def check(name, ok, detail=""):
    print("  %-52s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fail.append(name)

print("=" * 74)
print("Reproducibility package self-check")
print("=" * 74)

# --- 1. no broken symlinks --------------------------------------------------
broken = [os.path.relpath(os.path.join(d, f), ROOT)
          for d, _, fs in walk(ROOT) for f in fs
          if os.path.islink(os.path.join(d, f)) and not os.path.exists(os.path.join(d, f))]
broken += [os.path.relpath(os.path.join(d, s), ROOT)
           for d, ss, _ in walk(ROOT) for s in ss
           if os.path.islink(os.path.join(d, s)) and not os.path.exists(os.path.join(d, s))]
check("no broken symlinks", not broken, ";".join(broken[:3]))

# --- 2. no build junk -------------------------------------------------------
junk = []
for d, ss, fs in walk(ROOT):
    if "__pycache__" in ss: junk.append("__pycache__")
    junk += [f for f in fs if f.endswith((".pyc", ".DS_Store"))]
check("no __pycache__ / .pyc / .DS_Store", not junk, "%d found" % len(junk))

# --- 3. the certified checksum manifest verifies ---------------------------
man = os.path.join(ROOT, "model", "MD5SUMS_certified.txt")
bad = []
if os.path.exists(man):
    for line in open(man):
        line = line.strip()
        m = re.match(r"^([0-9a-f]{32})\s+\*?(.+)$", line)
        if not m: continue
        want, rel = m.group(1), m.group(2)
        p = os.path.join(ROOT, "model", rel)
        if not os.path.exists(p): bad.append(rel + " (missing)"); continue
        if hashlib.md5(open(p, "rb").read()).hexdigest() != want: bad.append(rel)
    check("model/MD5SUMS_certified.txt verifies", not bad, "%d mismatched" % len(bad))
else:
    check("model/MD5SUMS_certified.txt present", False)

# --- 4. the certifying hash of the solver module ---------------------------
m11 = os.path.join(ROOT, "model", "src", "dtpm", "modules", "m11_plasma_chemistry.py")
CERT = "7c50d0dedff3c2cfaabbd17d1024fb9d"          # the solver source shipped here; the
# archived run records under data/*/code_md5.txt keep the checksum of the copy
# that executed, which differs from this one in comments and identifier names only.
got = hashlib.md5(open(m11, "rb").read()).hexdigest() if os.path.exists(m11) else "-"
check("solver m11 matches the certified md5", got == CERT, got)

# --- 6. no machine-local paths or personal identifiers in shipped files -----
# Matched by digest for the same reason as check 12: a checker that spells out the
# names it looks for publishes them itself. Home-directory prefixes are matched
# structurally, the account and host names by hash of the lowercased token.
_PERSONAL_DIGESTS = frozenset((
    '1f18135415fce41e', '26b47209413e5776', '1adb9f794689b589',   # account, device
    '8cc7edf34f3c1710',                                            # allocation
))
# The facility and its university are named in the paper's own acknowledgements and appear in
# legitimate bibliographic citations, so the bare names are not a disclosure. What does leak
# infrastructure is a machine address, so those are matched as hostnames instead.
_ORGHOST = re.compile(r"(?<![A-Za-z0-9.-])[A-Za-z0-9-]+\.(?:[A-Za-z0-9-]+\.)*"
                      r"(?:ncsa|illinois)\.edu(?![A-Za-z0-9])")
_HOMEDIR = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\Users\\\\)([A-Za-z0-9._-]+)")
_TOKEN6 = re.compile(r"[A-Za-z][A-Za-z0-9]{5,}")
# Compute-node names are too short for the token matcher above (cn093, gpua031),
# so they are matched structurally instead.
_NODENAME = re.compile(r"(?<![A-Za-z0-9])(?:cn|gpua|nid)\d{3,}(?![A-Za-z0-9])")
_SCAN_EXT = (".py", ".sh", ".sbatch", ".md", ".json", ".yaml", ".yml", ".txt", ".tex", ".csv")

def _personal_hits(text):
    if _NODENAME.search(text):
        return True
    for m in _ORGHOST.finditer(text):
        # a published archive link is a citation, not infrastructure
        if not m.group().startswith(("www.ideals.", "ideals.")):
            return True
    for m in _HOMEDIR.finditer(text):
        user = m.group(1)
        if user not in ("shared", "runner", "opt"):
            return True
    for m in _TOKEN6.finditer(text):
        if hashlib.sha256(m.group().lower().encode()).hexdigest()[:16] in _PERSONAL_DIGESTS:
            return True
    return False

abs_hits = []
for dd, _, fs in walk(ROOT):
    for f in fs:
        rel = os.path.relpath(os.path.join(dd, f), ROOT)
        if rel in ("verify_package.py", "TREE_SHA256.txt"): continue
        # the path itself counts: a node name in a filename is as much a disclosure
        # as one in the body, and that is exactly how one survived a previous pass
        if _personal_hits(rel):
            abs_hits.append(rel); continue
        if not f.endswith(_SCAN_EXT): continue
        if _personal_hits(open(os.path.join(dd, f), encoding="utf-8", errors="replace").read()):
            abs_hits.append(rel)
check("no personal, account or facility identifiers", not abs_hits, "; ".join(abs_hits[:3]))

# --- 7. documented paths exist ---------------------------------------------
docs_bad = []
DOCS = ["README.md", "FIGURE_DATA.md", "THIRD_PARTY_DATA.md",
        "model/README.md", "ml/README.md", "cluster/README.md"]
for doc in DOCS:
    p = os.path.join(ROOT, doc)
    if not os.path.exists(p): continue
    for rel in re.findall(r'`([A-Za-z0-9_][A-Za-z0-9_./-]*/[A-Za-z0-9_./-]*)`', open(p).read()):
        rel = rel.rstrip("/")
        if rel.startswith(("http", "-")) or " " in rel: continue
        base = rel.split("/")[0]
        if base not in ("model", "data", "ml", "cluster"): continue
        if "<" in rel or "*" in rel or "NN" in rel: continue   # placeholders
        if not os.path.exists(os.path.join(ROOT, rel)):
            docs_bad.append("%s: %s" % (doc, rel))
check("paths quoted in the docs exist", not docs_bad, "; ".join(docs_bad[:3]))


# --- 10. checkpoint count -------------------------------------------------
n_ckpt = sum(1 for d,_,fs in walk(os.path.join(ROOT, "ml")) for f in fs if f.endswith(".pt"))
check("115 surrogate checkpoints present", n_ckpt == 115, "%d found" % n_ckpt)

# --- 11. no duplicated trainer trees --------------------------------------
tr = [os.path.relpath(os.path.join(d,f), ROOT)
      for d,_,fs in walk(os.path.join(ROOT, "ml")) for f in fs
      if f in ("train_ensemble.py","train_species.py","species_loader.py")]
dup = [n for n,c in collections.Counter(os.path.basename(x) for x in tr).items() if c > 1]
check("no duplicated trainer copies", not dup, ",".join(dup))

# --- 12. confidentiality: no reactor-vendor identity anywhere -----------------
# The terms are matched by digest, not spelled out. A checker that lists the strings it
# bans becomes a disclosure vector itself the moment the repository is readable, which
# defeats the purpose. Word and adjacent-word-pair tokens are normalised, hashed, and
# compared against the digests below. The generic short token is assembled at runtime
# for the same reason. Compressed containers are opened before scanning (PDF Flate
# streams, zip members of .pt/.npz, gzip, HDF5 names/attrs) -- raw-byte scanning alone
# would miss a term inside them. The three-letter token is checked in text files only:
# in binary float data those three bytes occur by chance and yield false positives.
_BANNED_DIGESTS = frozenset(('33b7e0fcdd54d68f', '4edac158299c8c61', '717d10aa8ac7c384', 'd0adca2467eb2068'))
_SHORT_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])" + chr(84) + chr(69) + chr(76) + r"(?![A-Za-z0-9_])")
_LEGACY_KEY = re.compile(r"(?<![A-Za-z0-9_])" + chr(116) + chr(101) + chr(108) + r"_geom")
TEXT_EXT = (".py", ".md", ".txt", ".yaml", ".yml", ".json", ".sh", ".sbatch",
            ".tex", ".csv", ".cfg", ".ini", ".toml")

def _decompressed_views(raw, name):
    """Yield decoded text for a file's raw bytes AND anything compressed inside it.

    Raw-byte scanning alone misses text in a Flate-compressed PDF content stream, a zip
    member of a .pt/.npz, or a gzip payload -- so a banned term could ride along unseen.
    Every container is best-effort: an unreadable one yields nothing rather than raising.
    """
    yield raw.decode("latin-1", "ignore")
    lower = name.lower()
    if lower.endswith((".pt", ".npz", ".zip", ".whl")):
        import io, zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for m in z.namelist()[:400]:
                    yield m
                    try:
                        yield z.read(m).decode("latin-1", "ignore")
                    except Exception:
                        pass
        except Exception:
            pass
    if lower.endswith(".gz"):
        import gzip
        try:
            yield gzip.decompress(raw).decode("latin-1", "ignore")
        except Exception:
            pass
    if lower.endswith(".pdf") or raw[:4] == b"%PDF":
        import zlib
        pos = 0
        while True:
            i = raw.find(b"stream", pos)
            if i < 0:
                break
            j = raw.find(b"endstream", i)
            if j < 0:
                break
            blob = raw[i + 6:j].lstrip(b"\r\n")
            try:
                yield zlib.decompress(blob).decode("latin-1", "ignore")
            except Exception:
                pass
            pos = j + 9
    if lower.endswith((".h5", ".hdf5")):
        try:
            import h5py, io as _io
            with h5py.File(_io.BytesIO(raw), "r") as f:
                acc = []
                f.visit(acc.append)
                for k in acc[:2000]:
                    acc2 = f[k].attrs if hasattr(f[k], "attrs") else {}
                    for ak, av in list(acc2.items())[:50]:
                        acc.append("%s %s" % (ak, av))
                yield " ".join(str(x) for x in acc)
        except Exception:
            pass

def _digest_hit(text):
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    for i, w in enumerate(words):
        if hashlib.sha256(w.encode()).hexdigest()[:16] in _BANNED_DIGESTS:
            return True
        if i + 1 < len(words):
            pair = w + words[i + 1]
            if hashlib.sha256(pair.encode()).hexdigest()[:16] in _BANNED_DIGESTS:
                return True
    return False

hits = []
for d, dirs, fs in walk(ROOT):
    dirs[:] = [x for x in dirs if x != "__pycache__"]
    for f in fs:
        fp = os.path.join(d, f)
        if os.path.abspath(fp) == os.path.abspath(__file__):
            continue
        try:
            raw = open(fp, "rb").read()
        except OSError:
            continue
        bad = False
        for body in _decompressed_views(raw, f):
            if _digest_hit(body):
                bad = True
                break
            if f.endswith(TEXT_EXT) and (_SHORT_TOKEN.search(body) or _LEGACY_KEY.search(body)):
                bad = True
                break
        if bad:
            hits.append(os.path.relpath(fp, ROOT))
check("no reactor-vendor identity in any shipped file",
      not hits, "; ".join(hits[:4]) if hits else "")

# --- 13. cluster jobs: every referenced script and flag file resolves ---------
import glob as _glob
missing = []
for job in sorted(_glob.glob(os.path.join(ROOT, "cluster", "*", "*.sbatch"))):
    body = open(job, encoding="utf-8", errors="ignore").read()
    for var, default in re.findall(r'\$\{(DTPM_[A-Z_]+|PHASE2_ROOT):-([^}"]+)\}', body):
        d = default.strip()
        if d.startswith("/") or "$" in d:
            continue          # site path or shell-expanded: not checkable here
        cand = os.path.join(ROOT, "model", d)
        alt = os.path.normpath(os.path.join(ROOT, "model", d))
        if not (os.path.exists(cand) or os.path.exists(alt)):
            missing.append("%s -> %s" % (os.path.basename(job), d))
check("cluster job default paths resolve against model/",
      not missing, "; ".join(missing[:4]) if missing else "")


# --- 14. whole-payload drift check ------------------------------------------
# The certified MD5 manifest covers 43 files under model/. This package exists as several
# independent copies with nothing keeping them in step, so the remaining ~900 files need a
# check too. TREE_SHA256.txt is that check; see make_tree_manifest.py for the two exclusions.
_tm = os.path.join(ROOT, "TREE_SHA256.txt")
if os.path.exists(_tm):
    import subprocess as _sp
    _r = _sp.run([sys.executable, os.path.join(ROOT, "make_tree_manifest.py"), "--check"],
                 cwd=ROOT, capture_output=True, text=True)
    _last = (_r.stdout.strip().splitlines() or [""])[-1]
    check("whole-payload manifest verifies", _r.returncode == 0, _last)
else:
    check("whole-payload manifest verifies", False, "TREE_SHA256.txt missing")

# --- 15. presentation: no box-drawing characters in source comments ----------
# The tree uses ASCII separators throughout. Unicode box-drawing characters render
# the same in most editors but are inconsistent with the rest of the source and are
# a common artefact of copy-and-paste, so this check keeps the style uniform.
_BOX = set("\u2500\u2550\u2501\u2502\u2503\u250c\u2510\u2514\u2518\u251c\u2524")
box_hits = []
for dd, _, fs in walk(ROOT):
    for f in fs:
        if not f.endswith((".py", ".sh", ".sbatch", ".md")): continue
        rel = os.path.relpath(os.path.join(dd, f), ROOT)
        if rel in ("verify_package.py", "TREE_SHA256.txt"): continue
        try:
            body = open(os.path.join(dd, f), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if _BOX & set(body):
            box_hits.append(rel)
check("ASCII separators only, no box-drawing glyphs", not box_hits, "; ".join(box_hits[:3]))

# --- every figure named in FIGURE_DATA.md exists, and its data with it --------
fig_bad = []
_fd = os.path.join(ROOT, "FIGURE_DATA.md")
if os.path.exists(_fd):
    seen = set()
    for line in open(_fd, encoding="utf-8"):
        if not line.startswith("| ") or line.startswith("| figure") or set(line.strip()) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        seen.add(cells[0])
        for col in (cells[2], cells[3]):
            for p_ in re.findall(r"data/[A-Za-z0-9_./*-]+", col):
                if "*" in p_:                       # a glob: check the directory holding it
                    p_ = os.path.dirname(p_)
                if p_ and not os.path.exists(os.path.join(ROOT, p_.rstrip("/"))):
                    fig_bad.append("data path %s" % p_)
    # every manuscript figure must be accounted for, schematics included
    expected = {str(n) for n in range(1, 28)} | {"S1"}
    missing = sorted(expected - seen, key=lambda s: (s == "S1", s))
    if missing:
        fig_bad.append("no row for figure(s) " + ", ".join(missing))
    check("FIGURE_DATA.md accounts for every figure", not fig_bad, "; ".join(fig_bad[:3]))
else:
    check("FIGURE_DATA.md accounts for every figure", False, "FIGURE_DATA.md missing")

# --- no internal version labels in shipped text ------------------------------
import re as _re
# Internal generation labels. Underscore counts as a separator so results_v7 and
# mesh_convergence_v2 are caught; a preceding digit or dot suppresses the match, so
# SF6, figure06a and an external version such as v10.6 are not.
_VER = _re.compile(r"(?:^|[^A-Za-z0-9.])(?:[vV][2-9]|6[a-d]|V6R)(?:[^A-Za-z0-9.]|$)")
ver_hits = []
for dd, _, fs in walk(ROOT):
    # ml/ is exempt. The trainer names its architectures and recipes with short tags
    # ("E1_v4_recipe", "surrogate_lxcat_v4_arch"); some are dictionary keys that select a
    # model, and the archived run records carry the same tags. They identify a training
    # recipe rather than a version of this repository, and renaming them would either
    # change behaviour or falsify the records. Everything a reader navigates by --
    # documentation, data, cluster jobs and the solver -- is checked strictly.
    if os.path.relpath(dd, ROOT).split(os.sep)[0] == "ml":
        continue
    for f in fs:
        if not f.endswith((".py", ".md", ".sbatch", ".json", ".yml", ".yaml", ".txt", ".csv")):
            continue
        rel = os.path.relpath(os.path.join(dd, f), ROOT)
        if rel in ("TREE_SHA256.txt", "verify_package.py"):
            continue
        try:
            body = open(os.path.join(dd, f), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if _VER.search(body) or _VER.search(rel):
            ver_hits.append(rel)
check("no internal version labels", not ver_hits, "; ".join(ver_hits[:3]))

# --- the plotted values must actually contain values ---------------------------
# Resolving a path proves a file exists, not that it holds anything. A CSV of nothing
# but commas passed every other check here, so this one reads the cells: each file needs
# at least one data row, and no column may be entirely empty.
import csv as _csv
empty = []
_pv = os.path.join(ROOT, "data", "plotted_values")
for dd, _, fs in walk(_pv):
    for f in sorted(fs):
        if not f.endswith(".csv"): continue
        rel = os.path.relpath(os.path.join(dd, f), ROOT)
        try:
            rows = list(_csv.reader(open(os.path.join(dd, f), encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            empty.append("%s unreadable" % rel); continue
        if len(rows) < 2:
            empty.append("%s has no data rows" % rel); continue
        ncol = len(rows[0])
        for c in range(1, ncol):
            col = [r[c].strip() for r in rows[1:] if len(r) > c]
            if col and all(v in ("", "nan") for v in col):
                empty.append("%s column %r is empty" % (rel, rows[0][c][:30])); break
check("plotted values are populated", not empty, "; ".join(empty[:3]))

print("=" * 74)
print("%d check(s) failed" % len(fail) if fail else "all checks passed")
sys.exit(1 if fail else 0)
