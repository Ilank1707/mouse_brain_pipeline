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

## Output structure

Each run is saved separately:

```text
<work_dir>/candidates/runs/<run_name>/
├── all_candidates.csv
├── candidate_run_metadata.json
├── coordinate_exports/
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

## Scientific safeguards

- Green and red are separate biological signal channels.
- Neither channel is used as background for the other.
- Preliminary rules are only review categories.
- Candidate counts are not final cell counts.
- Injection-mask assignments remain provisional until validated.
- The pipeline does not tune itself toward an expected number of cells.
- Raw TIFF files are read-only.
- Section 70 alone is not a whole mouse brain.

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
