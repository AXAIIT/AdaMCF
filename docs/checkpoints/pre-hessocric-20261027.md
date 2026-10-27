# Source checkpoint before HESSO-CRIC adaptation

Date: 2026-10-27 (Asia/Shanghai).

This checkpoint preserves the existing AIStation AdaMCF code before any new
HESSO-CRIC or Phi adaptation. It does not implement either adaptation.

- Repository: AXAIIT/AdaMCF.
- Parent published commit: `d6f7108e70fff713a9c9b452aed2d356a47756ab`.
- Checkpoint branch: `checkpoint/pre-hessocric-20261027`.
- Stable tag: `pre-hessocric-20261027`.
- Working source snapshot: 641 files, including all 400 Python files,
  shell scripts, model/configuration files, required tokenizer definitions and
  source documentation. All 37 core package Python modules are included.
- Existing published repository files, README, license, ignore rules, attributes
  and legacy reference examples are retained. The working server paths such as
  `test/pretrain_*` and `example/` are preserved as captured.
- Model weights, datasets, training logs, generated images, tensor dumps,
  checkpoints and other runtime outputs are not added by this commit.

The core optimizer/graph/transform source was unchanged throughout the preceding
read-only review. This snapshot preserves the current AdaMCF-modified HESSO,
existing HESSOCRIC and LoRA implementations; no optimizer was replaced by an
upstream version. Phi/CRIC adaptations remain future work.

## Verification scope

The complete selected-file SHA-256 inventory matched before and after capture.
All captured file bytes were verified after transfer. All 400 Python files passed
syntax parsing. The existing `.gitattributes` LF normalization is applied by Git;
the companion manifest records both original SHA-256 and committed Git blob SHA.
This checkpoint operation does not rerun model training or claim new accuracy.

The companion `pre-hessocric-20261027.sources.json` lists the complete captured
source set, modes and hashes. Previously published files outside that source set
are inherited unchanged from the parent commit. No inference/training environment
or dataset directory is placed in Git.

Use a separate development branch for subsequent adaptation so this tag remains
the identifiable pre-change source baseline.
