# Beat Saber Auto-Mapper Agent Guide

## Purpose
Build a C# solution that learns from high-quality Beat Saber maps and generates new playable Beat Saber maps from source audio.

The generator must not behave like a naive beat placer. It must produce maps that follow rhythm, structure, phrasing, emphasis, readability, and especially parity/flow.

This project uses a hybrid system:
- deterministic audio analysis
- supervised or sequence-based learning where useful
- rule-based constrained generation and repair
- strong validation

Do not build this as a pure end-to-end black box.

---

## Solution Shape
Keep the solution relatively compact.

Preferred project layout:

```text
/src
  BeatSaber.AutoMapper
  BeatSaber.AutoMapper.Cli
/tests
  BeatSaber.AutoMapper.Tests
/docs
  agent.md
  architecture.md
  cli-contract.md
  sqlite-schema.md
```

### Project responsibilities

#### BeatSaber.AutoMapper
Single core engine project containing most of the implementation:
- beatmap parsing and export
- internal canonical map model
- audio analysis
- dataset ingestion
- training logic
- feature extraction
- map generation
- validation
- persistence helpers
- diagnostics

This project may contain internal folders and namespaces for structure, but does not need to be split into many separate projects.

#### BeatSaber.AutoMapper.Cli
Thin command-line host:
- ingest dataset
- analyze audio
- train model
- evaluate model
- generate beatmap
- validate beatmap
- package map output

#### BeatSaber.AutoMapper.Tests
Unit and integration tests.

---

## Architecture Principle
Keep the solution compact, but keep the runtime pipeline structured.

The codebase does not need DDD, CQRS, or strict separation-heavy architecture.

However, the implementation must still preserve these logical stages:
1. import and normalize map data
2. analyze audio
3. extract training features
4. train model artifacts
5. generate candidate events
6. decode with flow and parity constraints
7. validate and repair
8. export Beat Saber map files

A compact solution layout is acceptable.
A chaotic runtime pipeline is not.

---

## Primary Goal
Generate Standard mode Beat Saber maps from source audio in a way that feels intentional and playable.

The differentiator is not merely being on beat.
The differentiator is producing maps with:
- coherent swing flow
- good parity
- comfortable movement
- musical emphasis
- stable difficulty
- low awkwardness

---

## Initial Scope
Start with a realistic v1.

### Must support
- C#/.NET implementation
- MP3 input
- OGG output packaging
- Beat Saber map parsing
- Beat Saber map export
- Standard mode
- Hard / Expert / ExpertPlus first
- notes first
- optional bombs later
- parity-aware generation
- validation and repair

### Delay until later
- full obstacle logic
- arcs and chains unless straightforward
- Easy and Normal generation
- mapper-style imitation
- UI/editor integration
- lightshow generation

---

## Core Strategy
Use a hybrid generator.

### Stage 1: Audio analysis
Analyze audio into a robust timing representation.

Must include:
- BPM estimation
- beat tracking
- downbeat estimation where possible
- onset detection
- energy envelope
- rhythmic intensity
- section segmentation
- candidate timing grid generation

The system must avoid placing notes on approximate rhythm only.
It must align events to real musical structure.

### Stage 2: Candidate event proposal
At each candidate subdivision, estimate whether a note or event should exist.

This can be done using:
- heuristics
- ML classifiers
- sequence model outputs
- or a combination

The first baseline does not need to be fully learned.
A strong deterministic timing engine plus learned placement is acceptable.

### Stage 3: Note attribute prediction
For each chosen event, predict or choose:
- hand/color
- lane
- row
- cut direction
- emphasis level
- pattern role

### Stage 4: Constrained decoding
Do not greedily accept note predictions.
Use a constrained search process that tracks hand state and rejects bad transitions.

Preferred options:
- beam search
- bounded best-first decoding
- score-and-prune candidate expansion

### Stage 5: Validation and repair
After generation, run the map through validation.
If possible, automatically repair:
- parity breaks
- awkward resets
- vision blocks
- density spikes
- lane imbalance
- obvious strain issues

---

## Canonical Internal Model
Never build logic directly on raw Beat Saber JSON structures.

Normalize imported maps into a canonical internal representation.

The canonical model should contain:
- song metadata
- bpm/timing information
- difficulty metadata
- note events
- bomb events
- obstacle events
- optional arc and chain support hooks
- beat position
- subdivision
- section label
- emphasis class
- local density
- recent swing context
- derived parity state

All training, validation, and generation should operate on this canonical model.
Export to Beat Saber schema only at the end.

---

## Data Strategy
The model is only as good as the dataset.

### Training corpus
Use a large corpus of community maps, but do not trust all maps equally.

### Filter aggressively
Build a map quality score using:
- community quality signals where available
- validation errors
- parity break rate
- timing sanity
- note density sanity
- duplicate detection
- schema validity
- exclusion of gimmick or malformed maps

### Initial dataset target
Prefer:
- Standard mode
- Hard / Expert / ExpertPlus
- maps that are considered reasonably playable

Avoid mixing everything into one undifferentiated corpus.

---

## Learning Strategy
The project should support learning, but learning should not own the entire problem.

### First baseline
Implement a hybrid baseline using:
- deterministic audio timing
- extracted candidate time grid
- simple placement model or heuristic scorer
- note attribute prediction model or heuristic selection
- parity-aware decoder
- repair pass

### Later improvements
After the baseline works, add:
- sequence model for pattern continuity
- phrase-aware generation
- style conditioning
- section-aware motif reuse

### Acceptable model stack
Implementation can use any practical C#-friendly stack.
Examples:
- ML.NET
- ONNX Runtime
- TorchSharp

Use whatever produces the best maintainable result.
Do not add complexity just to make the model more fashionable.

---

## Most Important Rule: Flow and Parity
Parity is not optional.

The system must explicitly model swing flow per hand.

Track per hand:
- previous note
- previous cut direction
- expected forehand/backhand state
- current swing angle trend
- timing since last action
- lane/row displacement
- whether recovery/reset is reasonable

Classify transitions as:
- good flow
- acceptable
- recovery required
- parity break
- awkward reset
- high strain
- invalid

This logic must exist as deterministic code even if the model learns some of it implicitly.

Do not assume the model will magically learn comfort and playability well enough on its own.

---

## Difficulty Handling
Difficulty should not be created by post-scaling one identical map.

Generation must be conditioned on target difficulty.

The target difficulty should influence:
- note density
- row and lane usage
- diagonal usage
- burst frequency
- crossover allowance
- reset frequency tolerance
- pattern vocabulary
- emphasis density

Start with:
- Hard
- Expert
- ExpertPlus

Delay Easy and Normal until a later phase.

---

## Validation System
Validation is a core subsystem.
Not a side utility.

### Validators to implement
- schema validity
- timing consistency
- overlap/conflict checks
- parity break detection
- reset detection
- strain heuristics
- lane and row balance
- vision block detection
- density envelope checks
- difficulty consistency
- suspicious repetition detection

### Validator output
Each finding should include:
- beat/time
- severity
- rule name
- explanation
- possible auto-fix if available

### Use of validator
The validator is used for:
- filtering training data
- scoring generated maps
- improving repair logic
- regression testing

---

## Persistence
Persistence is allowed and recommended.
SQLite is acceptable.

Use persistence for:
- song metadata
- analysis cache
- dataset manifests
- training runs
- model artifact metadata
- generation logs
- validation reports

Do not overengineer storage.
If tensors or spectrogram blobs are large, store them as files and index them in SQLite.

---

## CLI Requirements
The CLI should be thin and practical.

Suggested commands:

```text
ingest-dataset
analyze-audio
train
evaluate
generate
validate
pack
```

The CLI should orchestrate the engine, not reimplement it.

---

## Folder Structure Inside Core Project
A single core project is fine, but organize it internally.

Suggested folder layout:

```text
BeatSaber.AutoMapper
  /Audio
  /Beatmap
  /Canonical
  /Training
  /Generation
  /Validation
  /Persistence
  /Diagnostics
  /Features
  /Utilities
```

Use namespaces that match the folders.

---

## Testing Guidance
Tests are useful and should exist from the beginning.

### Prioritize tests for
- map import/export correctness
- timing conversion correctness
- beat/subdivision calculations
- parity transition classification
- validation rule behavior
- repair rule behavior
- generation invariants

### Useful test types
- unit tests for parity and timing
- schema roundtrip tests
- integration tests for sample songs/maps
- regression tests for previously broken map patterns

### Do not skip
If there is only limited test coverage early on, prioritize:
1. parity rules
2. timing conversion
3. import/export roundtrips

---

## Recommended Implementation Order
Follow this order unless there is a strong reason not to.

### Phase 0
- create solution and projects
- add test project
- define canonical model
- implement map importers
- implement simple exporter

### Phase 1
- implement audio decode pipeline
- implement BPM/beat/onset analysis
- cache analysis results
- create debug outputs for timing inspection

### Phase 2
- implement parity and flow validator
- implement reset detection
- implement density and lane balance checks

### Phase 3
- implement dataset ingestion
- implement quality filtering
- implement feature extraction
- create train/validation/test split flow

### Phase 4
- implement baseline generator
- deterministic candidate timing grid
- simple event placement scoring
- simple attribute scoring
- constrained decoder

### Phase 5
- implement training and model artifact loading
- compare learned placement against heuristic baseline
- adopt learned parts only where they outperform baseline

### Phase 6
- add repair pass
- add packaging
- add final reporting

### Phase 7
- add advanced features only after notes are good

---

## Baseline First
Do not begin with the most advanced possible model.

A strong first baseline is:
- deterministic timing grid
- onset-aware candidate scoring
- simple note placement logic
- parity-aware hand assignment
- constrained decoding
- repair pass

This baseline provides:
- something measurable
- something debuggable
- a fallback if learned models behave badly

---

## Diagnostics
Diagnostics must exist early.

Useful outputs:
- beat/onset debug CSV
- candidate grid dumps
- parity trace logs
- validator reports in JSON/Markdown
- generated map comparison reports
- density graphs

If the generator makes bad maps, diagnostics should make it obvious whether the issue came from:
- bad audio timing
- bad placement scoring
- bad attribute prediction
- weak decoding
- missing repairs

---

## Performance Guidance
Performance matters, but correctness matters first.

Priority order:
1. correctness
2. reproducibility
3. debuggability
4. performance

Cache expensive work:
- decoded audio metadata
- beat tracking results
- spectrogram-derived features
- dataset manifests

---

## Coding Guidance
Use idiomatic C#.
Do not overabstract.
Do not build architecture for architecture’s sake.

### Preferred style
- simple classes and records where appropriate
- internal helper types for canonical data
- explicit naming over cleverness
- deterministic functions where possible
- avoid magic constants without explanation

### Avoid
- premature plugin systems
- excessive interfaces for internal-only code
- unnecessary multi-project fragmentation
- AI abstractions that hide basic control flow

---

## Hard Constraints
Do these.

- keep the project in C#
- keep the solution compact
- keep training capabilities inside the core engine if convenient
- keep CLI thin
- keep a test project
- normalize imported maps
- treat audio timing as foundational
- treat parity as foundational
- use validation as a core subsystem
- prefer hybrid generation over raw black-box generation

---

## Do Not Do These
- do not generate Beat Saber JSON directly from raw MP3 without intermediate analysis
- do not trust all community maps equally
- do not try to solve playability with ML alone
- do not skip a repair pass
- do not support every Beat Saber mechanic in v1
- do not let project structure become more complicated than the runtime problem itself
- do not optimize for elegance over actual map quality

---

## Definition of Success
The project is successful when it can:
- ingest and normalize a high-quality corpus of Beat Saber maps
- analyze a new song accurately
- generate a Standard mode map that follows the song rhythm and phrasing
- preserve reasonable parity and flow
- export a valid Beat Saber map package
- produce maps that are meaningfully better than naive auto-mappers

The standard for success is not technically valid.
The standard is playable, coherent, and musically aligned.
