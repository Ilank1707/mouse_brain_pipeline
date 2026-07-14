# Mouse Brain Cell-Detection Pipeline

A Python pipeline for detecting and reviewing fluorescent cell candidates in serial two-photon mouse-brain images.

The dataset contains two registered biological signal channels:

| Internal name | Biological signal |
|---|---|
| `green_signal` | Green fluorescent dye |
| `channel_2_signal` | Red fluorescent dye |

Both channels are analyzed independently. Neither channel is treated as background or autofluorescence. The red channel may be weaker because of photobleaching, but the pipeline does not force it to contain fewer detections.

## Current dataset

- Seven optical planes per physical section
- Z spacing: `6.0 µm`
- XY pixel size: approximately `1.004 µm`
- Physical section thickness: `42 µm`
- 16-bit grayscale TIFF files
- Stack order: `z, y, x`
- Current available section: `070`
- Example files: `section_070_01.tif` through `section_070_07.tif`

```python
VOXEL_SIZES = (6.0, 1.004, 1.004)
```

## Pipeline

```text
Cellfinder candidate generation
→ channel-specific injection-site handling
→ fixed-XY measurements across seven planes
→ manual candidate review
→ separate green and red classifiers
→ cell / artefact / uncertain / injection
```

The pipeline produces **provisional candidates**, not final cell counts. A candidate is countable only after human confirmation or classification by a validated model.

## Main features

- Audits and pairs the seven TIFF planes
- Uses Cellfinder for permissive 3D candidate generation
- Keeps every candidate in `all_candidates.csv`
- Measures the same XY position across all seven optical planes
- Creates channel-specific injection masks
- Supports manual injection-mask corrections
- Exports review patches and full-section QC images
- Assigns each 3D candidate to one peak plane to avoid double counting
- Produces separate coordinate CSVs by candidate status
- Stores every run in an isolated output folder
- Never modifies the raw TIFF files

## Installation

Create and activate a virtual environment:

```powershell
cd C:\Users\saleem_lab\mouse_brain_pipeline
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Install Cellfinder if needed:

```powershell
pip install cellfinder
```

## Configuration

Create your local configuration:

```powershell
Copy-Item config.example.yml config.yml
```

Edit these paths in `config.yml`:

```yaml
data:
  green_signal_dir: "PATH_TO_GREEN_CHANNEL"
  channel_2_signal_dir: "PATH_TO_RED_CHANNEL"
  work_dir: "PATH_TO_WORK_DIRECTORY"
```

`config.yml` is ignored by Git because it contains machine-specific paths.

## Run the tests

```powershell
python -m pytest -q
```

Do not continue if tests fail.

## Run section 70

```powershell
$run = "section070_" + (Get-Date -Format "yyyyMMdd_HHmmss")

python scripts\run_candidate_pilot.py `
  --config config.yml `
  --first-section 70 `
  --n 1 `
  --run-name $run `
  --save-review-patches `
  --render-seven-planes
```

Do not add `--crop` when processing the full XY section.

This processes one physical section containing seven optical planes. It is not a whole-brain count.

### Full-QC vs. fast iteration

The command above is the **full-QC** run: it renders the seven-plane images and the
native full-resolution QC, saves review patches, runs the pair-correlation reports,
and writes the green/red cross-channel overlay.

For **faster iteration**, `--fast-qc` skips the seven-plane QC, the full-resolution
QC images and the review patches (the fast preview images, coordinate CSVs, and the
cross-channel overlay are still written):

```powershell
$run = "section070_" + (Get-Date -Format "yyyyMMdd_HHmmss")

python scripts\run_candidate_pilot.py `
  --config config.yml `
  --first-section 70 `
  --n 1 `
  --run-name $run `
  --fast-qc
```

An explicit request always wins over `--fast-qc` (e.g. `--fast-qc --render-seven-planes`
still renders the seven-plane images). Individual outputs can also be skipped on their
own:

| Flag | Skips |
|---|---|
| `--fast-qc` | seven-plane QC + full-resolution QC + review patches (unless explicitly requested) |
| `--skip-seven-plane-qc` | the seven-plane QC images |
| `--skip-fullres-qc` | the native full-resolution QC images |
| `--skip-review-patches` | the review patches |
| `--skip-pair-correlation` | the pair-correlation reports (alias of `--skip-spatial-analysis`) |
| `--skip-channel-overlay` | the green/red cross-channel overlay |

None of these flags change any candidate, status, mask, coordinate, or count.

## Cross-channel green/red overlay

After detection, every candidate in `all_candidates.csv` is measured with the **same
fixed-XY seven-plane measurement** in **both** biological channels (green
`green_signal` and red `channel_2_signal`) and labelled from the *measured* signal:

```text
green_dominant | red_dominant | both | unclear
```

This is audit only — it never changes a candidate, status, mask, or count, never uses
one channel as the other's input, and never forces the red channel to have fewer
detections. Outputs are written to `<run-dir>\channel_overlay\`:

```text
channel_overlay_candidate_measurements.csv   green/red peak, local background, SNR, ratio, dominant_channel
channel_overlay_summary.csv                  dominant_channel tallies (per channel + overall)
green_red_overlay_qc.png                     green/red composite, markers coloured by dominant_channel
```

Open the overlay QC image:

```powershell
explorer "PATH_TO_RUN\channel_overlay\green_red_overlay_qc.png"
```

To (re)build the overlay for an existing run without re-detecting:

```powershell
python scripts\overlay_channels.py --config config.yml --run-dir "PATH_TO_RUN"
```

## Output structure

Each run is saved separately:

```text
<work_dir>/candidates/runs/<run_name>/
├── all_candidates.csv
├── candidate_run_metadata.json
├── coordinate_exports/
├── channel_overlay/
├── qc/
├── review_patches/
└── seven_plane_qc/
```

Important coordinate exports include:

```text
all_candidate_coordinates.csv
preliminary_pass_coordinates.csv
preliminary_fail_coordinates.csv
manual_review_coordinates.csv
invalid_measurement_coordinates.csv
suspect_injection_coordinates.csv
confirmed_injection_coordinates.csv
confirmed_cell_coordinates.csv
unassigned_peak_plane.csv
```

`preliminary_pass_coordinates.csv` does not contain confirmed cells. Confirmed cells belong only in `confirmed_cell_coordinates.csv`.

## Seven-plane QC

The main seven-plane reports assign every 3D candidate to exactly one plane using its fixed-XY peak Z index:

```text
plane_01_peak_assigned_qc.png
...
plane_07_peak_assigned_qc.png
```

This prevents the same candidate from being counted more than once.

Optional support views may show one candidate on multiple planes. Their counts must not be added together.

## Manual review

```powershell
python scripts\review_candidates.py `
  --config config.yml `
  --channel green_signal `
  --candidates "PATH_TO_RUN\all_candidates.csv" `
  --labels "PATH_TO_RUN\manual_labels.csv" `
  --filter all `
  --limit 100
```

Use `channel_2_signal` to review the red channel.

Controls:

```text
1 = cell
2 = artefact
3 = injection
4 = uncertain
5 = skip
left/b = previous
right/n = next
up/down = change plane
r = raw/background-corrected
o = overlay
q = quit
```

## Validation and weighted calibration workflow

Follow these steps **in order**. Do not skip ahead to threshold changes.

1. **Run candidate generation** on the section.

   ```powershell
   python scripts\run_candidate_pilot.py --config config.yml `
     --first-section 70 --n 1 --run-name $run `
     --save-review-patches --render-seven-planes
   ```

2. **Inspect the seven-plane injection-mask QC.** Open `<run>\seven_plane_qc\`
   and `<run>\qc\` and confirm the injection core / analysis-exclusion masks
   (`03_injection_mask.png`, `11_injection_seed_filtering.png`) look correct
   before trusting any inside/outside split.

3. **Build the reason-stratified validation batch.** Green and red are sampled
   separately; failures are stratified by `preliminary_rule_reason`, and every
   row is written with an explicit inverse-probability `sample_weight`.

   ```powershell
   python scripts\make_validation_batch.py --config config.yml `
     --run-dir "<run>" --section 70 --out-dir "<val>"
   ```

4. **Label the validation patches.** Open `<val>\validation_review_patches\` and
   fill the blank `human_label` column in `validation_review_batch.csv` with one
   of `cell`, `artefact`, `uncertain`, or `injection`. The tool never labels for
   you.

5. **Run weighted calibration.** It reports both the raw unweighted confusion
   counts and inverse-probability **weighted** precision/recall/F1 so the balanced
   review sample is correctly scaled back to the full population.

   ```powershell
   python scripts\calibrate_candidate_rules.py --config config.yml `
     --run-dir "<run>" --batch "<val>\validation_review_batch.csv" `
     --out "<val>\calibration"
   ```

   `metrics_weighting` is `inverse_probability` when trustworthy weights exist. A
   legacy batch without weights still runs but is clearly flagged
   `unweighted_legacy_batch` and its numbers are never presented as population
   estimates.

6. **Inspect the false-positive and false-negative examples** in
   `false_positive_examples.csv` and `false_negative_examples.csv` (with their
   review patches). Also run the read-only audits:

   ```powershell
   python scripts\audit_candidate_generation_sources.py --run-dir "<run>"
   python scripts\audit_run_consistency.py --run-dir "<run>"
   ```

7. **Only then consider threshold changes.** `proposed_config_changes.yml` is
   REVIEW ONLY and is never applied. If you change a threshold, do it by hand and
   document old → new and the measured effect. Never tune toward a candidate or
   cell count.

8. **Rerun on an independent section** before claiming validation. Section 070
   alone cannot establish general performance.

Throughout: `preliminary_rule_pass` is **not** a cell, candidate totals are **not**
final cell counts, and the QC display settings never change any measurement or
detection.

## Spatial analysis

Two different spatial analyses exist. Pick the one you actually mean.

### Candidate-to-candidate pair correlation (default)

Measures candidate-to-candidate clustering. The normal
`scripts\run_candidate_pilot.py` command now runs it automatically; there is no
second command. Use `--skip-spatial-analysis` only when the reports are not wanted
for that run.

For every processed section, the run writes eligible reports beneath:

```text
<run-dir>\spatial_analysis\pair_correlation\section_070\
  green_signal\
    preliminary_pass\
    preliminary_fail\
    all_outside_injection\
    manual_review\
  channel_2_signal\
    ...same four statuses...
```

Each completed status folder contains `pair_correlation_g_r.png`,
`pair_density_per_mm2.png`, `pair_correlation_values.csv`,
`pair_density_values.csv`, and `metadata.json`. The main graph shows observed
`g(r)`, a dashed CSR line at 1, and the shaded 95% CSR envelope through 500 µm,
using 99 simulations by default. `manifest.csv` and `summary.json` at the
pair-correlation root list completed, low-count/crop-skipped, and failed reports.
Sections are analyzed separately and channel folders are never duplicated.

Cropped runs are skipped by default because crop boundaries bias clustering. A
failure in one report is recorded without stopping other channel/status reports.
The analysis does not change candidates, statuses, coordinates, masks, or counts.

The backward-compatible defaults are:

```yaml
postrun_spatial_analysis:
  enabled: true
  pair_correlation:
    enabled: true
    maximum_distance_um: 500
    simulations: 99
    random_seed: 20260713
  candidate_size_distributions:
    enabled: false
```

The legacy `candidate_size_distributions.png` is generated only when
`candidate_size_distributions.enabled` is explicitly set to `true` in the config.

### Injection-centred radial analysis (separate, optional)

A **different** analysis: candidate distance from the injection centre — not
candidate-to-candidate. Use only when that is what you want:

```powershell
python scripts\injection_centered_radial_analysis.py `
  --config config.yml `
  --run-dir "PATH_TO_RUN"
```

`scripts\radial_candidate_analysis.py` is a deprecated alias for this
injection-centred analysis and refuses to run without
`--confirm-injection-centered`.

## Scientific safeguards

- Green and red are separate biological signal channels.
- Neither channel is used as background for the other.
- Preliminary rules are only review categories: a `preliminary_rule_pass` is **not** a cell.
- Candidate totals are **not** final cell counts.
- Injection-mask assignments remain provisional until validated.
- The pipeline does not tune itself toward an expected number of candidates or cells.
- Raw TIFF files are read-only.
- QC display settings are brightness/contrast only — they never change any measurement or detection.
- Section 070 alone cannot establish general performance; validate on an independent section.

## Project structure

```text
mouse_brain_pipeline/
├── src/mouse_brain_pipeline/
├── scripts/
├── tests/
├── config.example.yml
├── runinstructions.txt
├── pyproject.toml
└── README.md
```

## Current development goals

- Improve injection-mask correction
- Reduce post-processing and rendering time
- Add radial candidate-density analysis
- Build representative manual labels
- Train and validate separate green and red 3D classifiers
- Later assign confirmed cells to mouse-brain atlas regions
