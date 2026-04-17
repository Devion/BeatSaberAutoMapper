# CLI Contract

## Command Shape

```text
bsam <command> [options]
```

`bsam` is an example executable name for `BeatSaber.AutoMapper.Cli`.

## Commands

### ingest-dataset
Import map archives and metadata into the local corpus catalog.

```text
bsam ingest-dataset --input <path> --library <path> [--recurse] [--copy-audio] [--db <path>]
```

Options:
- `--input <path>`: source folder containing map zips or unpacked folders
- `--library <path>`: normalized corpus storage location
- `--recurse`: scan subdirectories
- `--copy-audio`: copy linked audio into managed storage
- `--db <path>`: SQLite database path

Output:
- imported count
- skipped count
- malformed count
- duplicate count
- summary report path

### analyze-audio
Analyze one song and persist timing and feature cache.

```text
bsam analyze-audio --input <song.mp3> [--output <path>] [--db <path>] [--dump-debug]
```

Options:
- `--input <song.mp3>`: input audio file
- `--output <path>`: cache folder override
- `--db <path>`: SQLite database path
- `--dump-debug`: emit CSV/JSON debug files

Output:
- bpm estimate
- confidence
- beat count
- onset count
- cache artifact paths

### train
Run offline training for one or more model artifacts.

```text
bsam train --dataset <path> --profile <name> [--db <path>] [--artifacts <path>] [--epochs <n>]
```

Options:
- `--dataset <path>`: dataset manifest or folder
- `--profile <name>`: training profile name
- `--db <path>`: SQLite database path
- `--artifacts <path>`: model output path
- `--epochs <n>`: optional training epoch count if applicable

Output:
- artifact paths
- metrics summary
- validation metrics
- experiment identifier

### evaluate
Evaluate model artifacts against a held-out dataset.

```text
bsam evaluate --dataset <path> --artifacts <path> [--db <path>] [--report <path>]
```

Options:
- `--dataset <path>`: evaluation dataset
- `--artifacts <path>`: trained model folder
- `--db <path>`: SQLite database path
- `--report <path>`: report destination

Output:
- event precision/recall or equivalent metrics
- attribute accuracy metrics
- parity break statistics
- validator issue statistics

### generate
Generate a Beat Saber map from input audio.

```text
bsam generate --input <song.mp3> --title <name> --artist <name> --difficulty <Hard|Expert|ExpertPlus> --artifacts <path> --output <path> [--db <path>] [--cover <path>] [--dry-run] [--dump-debug]
```

Options:
- `--input <song.mp3>`: source song
- `--title <name>`: song title metadata
- `--artist <name>`: song artist metadata
- `--difficulty <...>`: target difficulty
- `--artifacts <path>`: trained artifact folder
- `--output <path>`: output folder
- `--db <path>`: SQLite database path
- `--cover <path>`: optional cover art
- `--dry-run`: skip packaging and only produce intermediate reports
- `--dump-debug`: emit detailed debug files

Output:
- generated map folder
- validation report
- repair report
- optional package zip

### validate
Validate an existing Beat Saber map folder or archive.

```text
bsam validate --input <path> [--db <path>] [--report <path>] [--json]
```

Options:
- `--input <path>`: map zip or unpacked map folder
- `--db <path>`: SQLite database path
- `--report <path>`: output report location
- `--json`: write JSON report in addition to text or markdown

Output:
- issue summary by severity
- issue breakdown by rule
- optional auto-fix candidates

### pack
Package an exported map folder into a distribution artifact.

```text
bsam pack --input <folder> --output <zip>
```

Options:
- `--input <folder>`: generated map folder
- `--output <zip>`: packaged archive path

Output:
- packaged zip path

## Exit Codes
- `0`: success
- `1`: validation or user input failure
- `2`: IO or external process failure
- `3`: training or generation pipeline failure
- `4`: unsupported map or audio format

## Logging
Minimum logging levels:
- quiet
- normal
- verbose
- diagnostic

Recommended global flags:
- `--verbosity <level>`
- `--db <path>`
- `--work <path>`
- `--no-color`

## Configuration
Allow command-line arguments to override config file values.

Recommended config file:

```text
bsam.json
```

Config categories:
- storage paths
- ffmpeg path
- default artifact path
- default difficulty profiles
- debug output preferences
- SQLite path
