#!/usr/bin/env python3
"""Write TREE_SHA256.txt: a whole-payload digest of this repository.

The certified MD5 manifest covers the 43 files under model/. That is the right scope for the
code provenance claim, but it leaves the data, the surrogate weights, the cluster scripts and
the documentation unguarded. This manifest closes that gap, so a reader can confirm the whole
tree arrived intact.

Excluded, because none of it is payload: this manifest itself, which cannot contain its own
digest; the .git directory, which is version-control bookkeeping and differs between any two
clones; and the files the operating system and the interpreter regenerate on their own,
namely .DS_Store, .pyc and __pycache__.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Everything in the tree is covered except the manifest itself, which cannot contain its
# own digest, and the two classes of file that the operating system and the interpreter
# regenerate on their own (.DS_Store from Finder, .pyc and __pycache__ from any import).
EXCLUDE_NAMES = {"TREE_SHA256.txt", ".DS_Store"}
EXCLUDE_RELPATHS = set()


def payload_files(root):
    for d, dirs, fs in os.walk(root):
        # .git is version-control bookkeeping, not payload, and its contents differ between
        # any two clones of the same tree.
        dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", ".git"))
        for f in sorted(fs):
            if f in EXCLUDE_NAMES or f.endswith(".pyc"):
                continue
            rel = os.path.relpath(os.path.join(d, f), root)
            if rel in EXCLUDE_RELPATHS:
                continue
            yield rel


def digest(root):
    per_file, whole = [], hashlib.sha256()
    for rel in payload_files(root):
        h = hashlib.sha256()
        with open(os.path.join(root, rel), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        per_file.append((h.hexdigest(), rel))
        whole.update(rel.encode("utf-8"))
        whole.update(h.digest())
    return per_file, whole.hexdigest()


def main():
    per_file, whole = digest(HERE)
    out = os.path.join(HERE, "TREE_SHA256.txt")
    if "--check" in sys.argv:
        if not os.path.exists(out):
            print("TREE_SHA256.txt missing"); return 1
        want = {}
        for line in open(out, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            h, _, rel = line.strip().partition("  ")
            want[rel] = h
        have = {rel: h for h, rel in per_file}
        missing = sorted(set(want) - set(have))
        added = sorted(set(have) - set(want))
        changed = sorted(r for r in set(want) & set(have) if want[r] != have[r])
        for label, items in (("missing", missing), ("unexpected", added), ("changed", changed)):
            for r in items[:10]:
                print("  %-11s %s" % (label, r))
        n = len(missing) + len(added) + len(changed)
        print("%d file(s) differ out of %d" % (n, len(want)))
        return 1 if n else 0
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Whole-payload SHA-256 manifest for this package.\n")
        fh.write("# Covers every file except this manifest, .git, .DS_Store and .pyc (see make_tree_manifest.py).\n")
        fh.write("# Verify with:  python3 make_tree_manifest.py --check\n")
        fh.write("# payload-digest %s\n" % whole)
        for h, rel in per_file:
            fh.write("%s  %s\n" % (h, rel))
    print("wrote TREE_SHA256.txt: %d files, payload digest %s" % (len(per_file), whole[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
