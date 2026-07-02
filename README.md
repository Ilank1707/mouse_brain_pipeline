# mouse_brain_pipeline

Audit, prepare, pilot-test and (eventually) analyse a **very large serial
two-photon mouse-brain dataset** made of **two registered signal channels**.

| Channel | Neutral name        | Notes |
|--------:|---------------------|-------|
| 1       | `green_signal`      | green fluorescent biological marker |
| 2       | `channel_2_signal`  | second Red fluorescent marker (possibly an injection on the opposite side) |

The real biological marker names are **unknown** — neutral names are used
throughout. Both channels are **signal**; they are counted **independently**.

> ### Critical scientific restriction
> The two supplied folders are **both signal channels**. The pipeline **never**
> silently uses one signal channel as the anatomical/autofluorescence
> *background* channel that Cellfinder/Brainmapper requires. Full Brainmapper
> execution stays **disabled** until you provide a real `background_dir` **and**
> confirm a BrainGlobe `orientation`. A built-in **experimental** 3D blob
> detector is provided for pilot testing only; its outputs are labelled
> **"candidate detections"**, never final cell counts.

The intended end-to-end analysis mirrors the Cellfinder / Brainmapper
whole-brain 3D cell-detection workflow (UCL whole-brain paper).

---

## Dataset facts (confirmed)

* Filename: `section_070_01.tif` → section `070`, optical plane `01` (valid 01–07).
* 7 optical planes per major physical section; **Z-spacing 6 µm**; physical cut
  **42 µm**; XY pixel ≈ **1.004 µm**.
* TIFFs ≈ **13,912 × 9,906**, **16-bit** grayscale, ≈ **250 MB** each, **> 1,000** files.
* Matching TIFFs across the two channels are already spatially aligned.

Relative-Z is computed as:

```
global_plane = (section - first_section) * 7 + (plane - 1)
z_um         = global_plane * 6
```

…and is only trusted after the audit confirms sections are **contiguous**.

---

## Safety guarantees baked in

* Raw TIFFs are **read-only**: never edited, renamed, moved, overwritten or recompressed.
* **No whole-brain / whole-dataset RAM loads** — reads are per-plane / per-tile, lazy
  (memory-mapped via `tifffile`; strided downsampling for previews).
* All generated files go to a separate `work_dir`. Full 16-bit data is preserved;
  downsampling is for previews only.
* `--dry-run`, logging, and resumable, restartable steps throughout.
* Start small: the pilot defaults to **two contiguous sections**.
* Brainmapper input paths must **not contain spaces** — the tools detect and warn.

---

## Installation

> **This machine right now:** Python **3.14** only, **no conda**, and none of the
> scientific packages are installed yet. The BrainGlobe/Cellfinder stack targets
> **Python 3.11**, so a dedicated 3.11 environment is required for full analysis.
> The commands below are **proposed** — review them; nothing is auto-installed.

### Option A — lightweight (audit / previews / pilot / candidate detection)

Works on Python 3.10/3.11. Does **not** include Cellfinder.

**Windows PowerShell**
```powershell
# Install a Python 3.11 (e.g. from python.org) then:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Option B — full stack incl. Brainmapper/Cellfinder (recommended: conda)

```bash
# Install miniforge/mambaforge first (provides conda). Then:
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate mouse-brain
pip install -e ".[brainmapper,viz]"
```

Verify your environment at any time:

```powershell
python scripts/audit_dataset.py --check-env
```

---

## Configure

```powershell
Copy-Item config.example.yml config.yml
# edit config.yml: set green_signal_dir, channel_2_signal_dir, work_dir
```

`background_dir` stays `null` and `registration.orientation` stays `null` until
you genuinely have them. Keep `work_dir` on a **space-free** path.

---

## Usage

### 1. Audit the dataset
```powershell
python scripts/audit_dataset.py --config config.yml --dry-run   # validate, write nothing
python scripts/audit_dataset.py --config config.yml             # writes manifest + summary
```
Outputs (in `work/audit/`): `manifest.csv`, `missing_files.csv`,
`dataset_summary.json`, `audit.log`. Exits **non-zero** on serious
structure/pairing errors.

### 2. Create previews
```powershell
python scripts/make_previews.py --config config.yml --n 5 --downsample 16
```
Per-channel PNGs + a two-channel overlay (`green_signal`→green,
`channel_2_signal`→magenta), percentile-scaled for display only.

### 3. Prepare the current one-section pilot
```powershell
python scripts/prepare_pilot.py --config config.yml --first-section 70 --n 1
```
Writes ordered file lists (and optional `--symlinks`); prints the exact planes
and the relative-Z range; **refuses** to continue if any plane is missing.

### 4. Run the provisional candidate detector (pilot only)
> **`--crop` is `X_MIN X_MAX Y_MIN Y_MAX`** (full-resolution pixels). Full planes
> are `9906 (H) × 13912 (W)`.

```powershell
# new ISOLATED full-section run (section 70) with the seven-plane QC images:
python scripts/run_candidate_pilot.py --config config.yml --first-section 70 --n 1 `
  --run-name section070_test_07 --save-review-patches --render-seven-planes
# single section-70 crop:
python scripts/run_candidate_pilot.py --config config.yml --first-section 70 --n 1 --crop 1000 5000 500 4000
# with centred manual-review patches:
python scripts/run_candidate_pilot.py --config config.yml --crop 1000 5000 500 4000 --save-review-patches
```

**Every real run gets its own isolated folder** under
`<work_dir>/candidates/runs/<run_id>` (`--run-name` sets `<run_id>`; otherwise it
is `<UTC-timestamp>_section<NNN>`). All of that run's files live there
(`all_candidates.csv`, `candidate_run_metadata.json`, `candidate_status_summary.csv`,
`coordinate_exports/`, `qc/`, `review_patches/`, `seven_plane_qc/`). A previous run
is **never** silently overwritten (a non-empty `--run-name` is refused), files are
never copied from another run, and `<work_dir>/candidates/latest_run.json` is
written **only after** a run completes successfully. The run metadata records the
code version, config path + hash, run name, timestamp, candidate + status counts,
per-channel source TIFF dimensions and the (none) classifier state, so attempts
can be compared without mixing files. `--render-seven-planes` renders the
peak-assigned QC images for **this run only** (never a "newest CSV" search).

Two **backends** (set `detection.backend`):
* `cellfinder_candidates` *(default, preferred)* — a thin adapter around the real
  `cellfinder.core.detect.detect.main` candidate-detection stage (no background
  channel needed). Requires `pip install cellfinder`; if absent the run **stops
  with an install message** (it never silently falls back).
* `pilot_log3d` — a self-contained rule-based 3D detector (no Cellfinder needed),
  selected explicitly.

The Cellfinder call uses `cellfinder.core.detect.detect.main`, with
`signal_array` and `voxel_sizes` in `z,y,x` order. It forwards the configured
`soma_diameter`, `max_cluster_size`, `ball_xy_size`, `ball_z_size`,
`ball_overlap_fraction`, `soma_spread_factor`, `log_sigma_size`,
`n_sds_above_mean_thresh`, `n_sds_above_mean_tiled_thresh`,
`tiled_thresh_tile_size`, `outlier_keep`, `artifact_keep`, `batch_size`, and
`torch_device`. Original Cellfinder centroid/type metadata are retained.

Shared post-processing for both backends: one **shared, permissive tissue mask**
built from **both** channels; separate per-channel injection **core** and larger
**analysis-exclusion** masks; fixed-XY per-plane contrast profiles; physical
size/morphology features; non-max suppression;
explicit coordinate bookkeeping (`x/y_local_px`, `x/y_global_px`, `z_index`,
`section_relative_z_um`, `global_z_um`) with hard validation.

Hand-rule states are sampling categories only: `preliminary_rule_pass`,
`preliminary_rule_fail`, `artifact`, and `manual_review`. Automatic mask hits
remain `suspect_injection_mask` until the mask passes QC and is explicitly
validated. `injection_site` means manual geometry, a validated automatic mask,
or a human injection label. Invalid measurements have the distinct display
state `invalid_measurement` and the sampling category `manual_review`.
Unreviewed preliminary candidates always have `included_in_count=false`.

Outputs (in `work/candidates/`):
* `all_candidates.csv` — every candidate carries fixed-XY plane contrasts,
  measurement validity, `current_status`, `included_in_count`,
  `rejection_reason`, candidate-generation provenance and Cellfinder z-mapping
  diagnostics;
* `preliminary_pass_candidates.csv` — the documented subset that passed the
  preliminary rules (still not cells);
* `review_batch.csv` — a stratified manual-review table with
  `review_sampling_category`;
* `candidate_status_summary.csv` — mutually exclusive status counts that
  reconcile exactly to all generated candidates;
* `candidate_run_metadata.json` — exact Cellfinder, acquisition, mask and crop
  parameters used for the run;
* `manual_labels.csv` — atomically updated by the review application
  (`cell` / `artefact` / `injection` / `uncertain`);
* `qc/shared_tissue_mask.png` and, per channel,
  `qc/<channel>_section_<NNN>/` with seven QC images
  (`01_raw_projection`, `02_shared_tissue_mask`, `03_injection_mask`,
  `04_candidates_before_injection_exclusion`,
  `05_candidates_after_injection_exclusion`, `06_manual_review_sample`,
  `07_candidate_interpretation_audit`);
* `review_patches/…` (centred, measurement/annulus circles, shared scaling,
  raw + background-corrected, and a fixed-XY Z profile) when
  `--save-review-patches` is given.

**These are PROVISIONAL CANDIDATE detections, not validated cell counts.** The
thresholds in `config.yml` are initial pilot values and must be tuned against
manually identified cells. If any **invalid coordinate** is detected the run
prints a prominent warning and **withholds scientific counts**.

To install Cellfinder for the preferred backend (clean Python 3.11 env):
```powershell
pip install cellfinder   # GPU recommended; set detection.cellfinder.torch_device: "cpu" otherwise
```

### 5. Review and label candidates (seven aligned planes)

```powershell
python scripts/review_candidates.py --config config.yml --channel green_signal
python scripts/review_candidates.py --config config.yml --channel channel_2_signal
```

Each candidate is inspected across **all seven aligned optical planes** of its
section. The seven planes load in order (`_01`..`_07`) and the **same global
`(x, y)`** window is cropped from every plane — planes are never independently
recentred. The window is built from the section's seven TIFFs only; raw values
are read, never written, and stay 16-bit until display scaling. The layout:

* a **seven-panel montage** (plane 1..7, identical display size and XY centre,
  candidate crosshairs, **peak plane** gold, detected **support planes** green);
* a **bg-corrected** montage row, so raw and background-corrected views are both
  available;
* a **larger single-plane view** with a Z slider and keyboard scrubbing;
* a **maximum-intensity projection**, clearly labelled *display aid only* — it
  must **not** be used alone to decide whether something is a cell;
* an **optional colour-coded Z overlay** (each plane a distinct hue), in addition
  to — not replacing — the montage; visualisation only, raw TIFFs unchanged;
* a **fixed-XY** central-intensity + local-contrast plot across the seven planes.

Keyboard: `1 cell`, `2 artefact`, `3 injection`, `4 uncertain`, `5 skip`,
`left`/`b previous`, `right`/`n next`, `up`/`]` next plane, `down`/`[` previous
plane, `r` toggle raw/bg-corrected, `o` toggle overlay, `q quit`. Every label is
written immediately to `manual_labels.csv` with columns `candidate_id, channel,
section, x_global_px, y_global_px, z_index, manual_label, reviewer, timestamp`.
An existing label is **never overwritten silently** (the previous label is shown
when revisiting; changing it requires `--allow-label-changes`). The two channels
are labelled and resumed **independently** (keyed by `candidate_id` + `channel`).
A candidate is never a final cell until it has a human label or a validated
classifier prediction. `--filter` supports `preliminary_rule_pass`,
`preliminary_rule_fail`, `near_threshold`, `single_plane`, `many_planes`,
`outside_injection`, `inside_injection`, `invalid_measurement`, and
`random_sample`. Review resumes with unlabelled candidates by default.

### 5b. Whole-section seven-plane candidate QC (full-resolution)

For a static, full-brain view of candidates on each optical plane (the same kind
of image as `04_candidates_before_injection_exclusion.png`, but rendered
separately for `section_070_01.tif` .. `section_070_07.tif`):

**Preferred: peak-assigned QC for one run directory (no double counting).** Each
candidate is a 3D object drawn on **exactly one** plane — its canonical peak
plane `fixed_xy_peak_z_index` (0-based: 0→plane 01, 6→plane 07) — so the seven
plane counts never double-count and `sum(planes 01–07) + unassigned = unique
candidates`. Candidates with a missing/invalid peak Z are not guessed; they go to
`coordinate_exports/unassigned_peak_plane.csv` and are reported.

```powershell
python scripts/render_seven_plane_qc.py `
  --config config.yml `
  --run-dir "C:\mouse_brain_work\candidates\runs\section070_test_07" `
  --channel green_signal --section 70
```

`--run-dir` reads **only** `<run-dir>/all_candidates.csv` and
`<run-dir>/candidate_run_metadata.json` (no folder search). It **refuses** when
the metadata is missing, the CSV is from a crop while a full section is rendered
(`--allow-cropped` to override), the section/channel do not match, the TIFF
dimensions disagree with the metadata, or the counts do not reconcile. For each
plane it writes `plane_0X_peak_assigned_native.png` (native resolution, lossless)
and `plane_0X_peak_assigned_qc.png` (native brain + **white header/footer** with
the title, count summary and a legend of per-plane counts, in the
`04_candidates_*` status colours/symbols). A `seven_plane_qc/support_views/`
folder additionally shows each candidate on **every** supported plane, clearly
warning "DO NOT SUM THESE PLANE COUNTS" — it is a visualisation, not a count.

The earlier whole-section overlay mode (every candidate on every plane, faint
where unsupported) is still available for quick comparison:

```powershell
# auto-select the newest FULL-section run for this channel+section
python scripts/render_seven_plane_qc.py `
  --config config.yml --channel green_signal --section 70 `
  --find-latest-full-section `
  --candidates-root "C:\mouse_brain_work\candidates" `
  --output "C:\mouse_brain_work\candidates\seven_plane_qc" `
  --marker-mode all
# emphasise only support planes:  --marker-mode support
# robust per-plane display (dim individual planes): --display-mode per-plane-robust
# Fiji-like channel-2 window (display only): --display-min 0 --display-max 513
```

This is **separate** from the interactive reviewer (it does not replace it). It
writes seven full-resolution lossless PNGs (`plane_01_candidates_fullres.png` ..
`plane_07_candidates_fullres.png`) **at the original TIFF width/height with no
resize**, a `seven_plane_candidate_montage.png`, a marker-free
`seven_plane_raw_montage.png`, and `seven_plane_qc_metadata.csv` (source TIFF
path, source/saved width+height, channel, section, optical plane, display
min/max, candidates displayed, candidates supported on that plane). Markers are
drawn directly at native resolution with Pillow using the existing status
colours and symbols. Each candidate's `(x_global_px, y_global_px)` is drawn at
the **same** position on every plane (registered planes share XY; never
recentred).

**Picking the right candidate CSV.** Before rendering it prints the CSV path,
the channel, the section, the number of candidates loaded, the run crop and the
source image dimensions, and it reads the sibling `candidate_run_metadata.json`.
If that metadata says the CSV came from a **cropped** run it **refuses** to
render a full section (use `--allow-cropped-candidates` to override). It also
**warns** when the loaded candidate count differs sharply from the count
recorded in the run metadata (the classic "old cropped run" mistake).
`--find-latest-full-section` searches `--candidates-root`
(default `C:\mouse_brain_work\candidates`) and selects the newest
`all_candidates.csv` whose metadata has crop = none, the section processed and
the channel present, choosing by the run-metadata **timestamp** (never by folder
modification time).

`--marker-mode all` (default) shows every candidate on every plane — unsupported
planes use a thinner, fainter marker and support/peak planes a stronger one, but
nothing is hidden; `--marker-mode support` shows a candidate only on its
`fixed_xy_support_z_indices` planes. `--display-mode configured` (default) uses
one section window from the channel's QC display config; `--display-mode
per-plane-robust` computes a robust window **independently per plane** (finite
in-tissue pixels, black background excluded, injection-core tail excluded from
the upper percentile) and records each plane's min/max. The display window is
**visualisation only** — it never reruns detection and never alters the raw
TIFFs, candidate coordinates, counts or statuses.

### 6. Annotate references and measure candidate-generation recall

```powershell
python scripts/annotate_reference_cells.py --config config.yml `
    --channel green_signal --section 70 --crop 1000 5000 500 4000
python scripts/evaluate_candidate_recall.py --config config.yml
```

Reference clicks are saved separately in `manual_reference_points.csv`; they do
not become final cells. Recall uses one-to-one matching against original
Cellfinder coordinates and reports nothing until at least one reference exists.
Unmatched Cellfinder candidates are not called false positives.

### 7. Train and apply separate single-channel classifiers

Training refuses to start until that channel has at least 50 manually labelled
cells and 50 artefacts (configurable):

```powershell
python scripts/train_candidate_classifier.py --config config.yml --channel green_signal
python scripts/train_candidate_classifier.py --config config.yml --channel channel_2_signal
```

The PyTorch input shape is `[N,1,Z,Y,X]`. One dye is never used as background
for the other. Splits are grouped by spatial tile by default. With section 70
alone, trustworthy independent section-level validation is not possible.

```powershell
python scripts/classify_candidates.py --config config.yml --channel green_signal `
    --model C:\mouse_brain_work\classifiers\green_signal\<version>\model.pt
python scripts/classify_candidates.py --config config.yml --channel channel_2_signal `
    --model C:\mouse_brain_work\classifiers\channel_2_signal\<version>\model.pt
```

The default states are `predicted_cell` at probability >= 0.80,
`predicted_artifact` at probability <= 0.20, and `manual_review` between them.
Predicted cells remain excluded from counts unless the model bundle is marked
validated by a separate scientific validation gate; training completion never
passes that gate. A human `cell` label is countable immediately.

### 8. Brainmapper dry run
```powershell
python scripts/run_brainmapper.py --config config.yml --dry-run
```
Prints the exact command and any **blockers** (missing background channel /
orientation / `brainmapper` CLI). It will **refuse** to run a signal-only
dataset. Only after you set a real `background_dir` and `orientation`:
```powershell
python scripts/run_brainmapper.py --config config.yml --confirm
```

### 9. Summarise validated results
```powershell
python scripts/summarize_results.py --config config.yml `
    --candidates work/candidates/classified_candidates_green_signal.csv
# optional PROVISIONAL spatial overlap (off by default):
python scripts/summarize_results.py --config config.yml `
    --candidates work/candidates/classified_candidates_green_signal.csv --overlap
```
Object-level + region-level CSVs. Overlap matching is one-to-one, configurable
and **disabled by default**; matches are **provisional spatial classifications**,
never "double-positive".

### 10. Review in Napari
```powershell
pip install "napari[all]"
napari
# Open a green_signal TIFF and the matching channel_2 TIFF as two layers
# (they are already aligned). Load a classified candidate CSV as a
# Points layer (columns z_plane, y_px, x_px) to inspect candidates.
```

---

## Tests

```powershell
pip install pytest
pytest -q
```

* `test_filenames.py` and `test_z_coordinates.py` are **pure standard-library**
  (run on any interpreter): leading zeros, numerical sorting, seven-plane sets,
  missing-plane logic, duplicate keys, Z geometry, contiguity.
* `test_manifest.py` builds **small synthetic TIFFs** (skips automatically if
  `numpy`/`tifffile` are absent): clean dataset, missing plane, duplicate plane,
  channel shape mismatch, unpaired plane, output files, and **tile-overlap
  duplicate removal** (NMS).
* `test_candidate_detection.py` builds **synthetic 3D stacks** (skips without
  `numpy`/`scipy`) and checks: only valid Z values (0–36 µm); no NaN/inf/
  billion-scale coordinates; crop-local→global conversion; full-image manual
  rectangles mapped into a crop; a broad injection region yields a non-empty mask
  and its candidates are excluded; the shared tissue mask covers dim tissue (not
  just the injection); compact objects in planes 1–2 / 2–4 are retained; a
  spatially jumping object is rejected; review patches are centred identically in
  every plane; and the Cellfinder adapter receives a `z, y, x` array with voxel
  sizes `(6.0, 1.004, 1.004)`.
* `test_candidate_workflow.py` checks the full noise fallback hierarchy,
  padding exclusion, canonical fixed-XY fields, injection-mask provenance and
  fail-safe status, atomic/conflict-safe labels, review CSV schema, grouped
  classifier splitting, validation limitations, reference-point matching, and
  raw-TIFF immutability.
* `test_review_montage.py` builds small synthetic 16-bit TIFFs and proves the
  seven-plane reviewer guarantees: planes load in order `_01`..`_07`; every
  patch uses the same XY centre; the peak plane is highlighted; the max
  projection never alters the raw stack; boundary candidates are zero-padded
  safely; labels resume from `manual_labels.csv` (exact 9-column schema); raw
  TIFFs are never written; and both biological channels label independently.
* `test_seven_plane_qc.py` proves the whole-section renderer: planes load
  `_01`..`_07`; each full-resolution PNG keeps the source TIFF dimensions and is
  never resized; candidate coordinates are identical across planes; support and
  peak planes are matched correctly; `support_only` hides unsupported planes; the
  raw arrays/TIFFs are never modified; and both channels render independently
  (including the Fiji-like 0-513 channel-2 window).
* `test_seven_plane_selection.py` proves the candidate-file selection: a cropped
  candidate CSV is rejected for a full-section render; the latest valid
  full-section CSV is selected by run-metadata timestamp (not folder mtime); the
  loaded candidate count is reported exactly; `--marker-mode all` shows every
  candidate on every plane while `--marker-mode support` shows only supported
  ones; and per-plane-robust display leaves the raw arrays/TIFFs and dimensions
  unchanged.
* `test_run_isolation_and_exports.py` proves run isolation and coordinate
  exports: every run gets a new isolated directory (no silent overwrite);
  `latest_run.json` is only written on the explicit success call; every valid
  candidate is assigned to exactly one peak plane and `assigned + unassigned =
  unique`; one-based `peak_optical_plane` is derived from the zero-based Z; a
  missing/invalid peak Z is reported as unassigned (never guessed); the
  per-status coordinate CSVs are produced; the confirmed-cell CSV excludes bare
  preliminary passes; status counts reconcile; and one section is not a brain.
* `test_seven_plane_report.py` proves the peak-assigned render: the renderer
  reads only the supplied run directory (never an older run); each candidate is
  drawn on exactly one main plane image; support views may repeat candidates but
  carry a DO-NOT-SUM warning; each QC image has a title, count summary and a
  legend whose counts match the candidates drawn on that plane; native images
  keep the source TIFF dimensions; the raw TIFFs are untouched; and the run-dir
  render refuses missing metadata, a cropped CSV, or a TIFF-dimension mismatch.

---

## Project layout

```
mouse_brain_pipeline/
├── src/mouse_brain_pipeline/   # library code (lazy heavy imports)
│   ├── filenames.py            # parsing, numeric sort, Z geometry (stdlib)
│   ├── config.py               # typed config from config.yml
│   ├── utilities.py            # env checks, logging, path safety (stdlib)
│   ├── audit.py                # discover / pair / validate / manifest
│   ├── previews.py             # downsampled QC previews
│   ├── pilot_stack.py          # contiguous pilot range prep (no data copy)
│   ├── candidate_detection.py  # multi-stage candidate pipeline (2 backends)
│   ├── cellfinder_adapter.py   # adapter around cellfinder.core.detect.detect.main
│   ├── candidate_qc.py         # QC images, review patches, review_batch.csv
│   ├── brainmapper_runner.py   # guarded Brainmapper command builder/runner
│   ├── overlap.py              # provisional one-to-one spatial overlap
│   └── summarize.py            # object/region CSV export
├── scripts/                    # thin CLIs for each step
├── tests/                      # synthetic-data tests
├── config.example.yml
├── environment.yml             # Python 3.11 full stack (conda)
└── pyproject.toml
```

## Scientific caveats

* `green_signal` / `channel_2_signal` are **neutral** names; the markers are unknown.
* Channel 2 is **not** assumed to be background/autofluorescence.
* Spatial overlap between channels is **provisional** and must be validated before
  any "double-positive"/co-localisation claim.
* The pilot blob detector is **experimental** — use Brainmapper/Cellfinder with a
  real background channel for defensible counts.
