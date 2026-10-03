# Documentation in Simplified Technical English

> This directory has the Simplified Technical English (ASD-STE100) version of the project documents. The original documents are in [`docs/`](..) and in the project [README.md](../../README.md).

Each file here has the same name as its original and the same sections, the same code blocks and the same numbers. Only the prose is different.
The files follow the STE writing rules with some freedom. They are not certified. [STYLE.md](STYLE.md) gives the rules and the project vocabulary.

| STE version | original | content |
|---|---|---|
| [project_readme.md](project_readme.md) | [README.md](../../README.md) | installation, use, test commands |
| [compilation_flow.md](compilation_flow.md) | [compilation_flow.md](../compilation_flow.md) | the stages from an OPS loop to a running kernel, with the IR of each stage |
| [loop_fusion.md](loop_fusion.md) | [loop_fusion.md](../loop_fusion.md) | lazy run, dependence analysis, the planner, the fusion rules |
| [cloverleaf.md](cloverleaf.md) | [cloverleaf.md](../cloverleaf.md) | CloverLeaf 2D and 3D through ops-mlir, and the checks of the result |
| [cloverleaf_large.md](cloverleaf_large.md) | [cloverleaf_large.md](../cloverleaf_large.md) | the large CloverLeaf decks on the A100 |
| [tgv_evaluation.md](tgv_evaluation.md) | [tgv_evaluation.md](../tgv_evaluation.md) | the Taylor-Green vortex evaluation on the development machine |
| [tgv_evaluation_a100.md](tgv_evaluation_a100.md) | [tgv_evaluation_a100.md](../tgv_evaluation_a100.md) | the Taylor-Green vortex evaluation on the A100 |

## Test the files

```bash
python3 docs/ste/ste_check.py --summary docs/ste/*.md
for f in compilation_flow loop_fusion cloverleaf cloverleaf_large tgv_evaluation tgv_evaluation_a100; do
  python3 docs/ste/ste_check.py --compare docs/$f.md docs/ste/$f.md
done
python3 docs/ste/ste_check.py --compare README.md docs/ste/project_readme.md
```

The first command shows the style counts. The other commands show that the code blocks, the number of headings and the numbers are the same as in the originals.
The program tests only the rules that a program can test. It does not find a wrong meaning. A person must read the text again.

## Known limits

- The check finds a few false alarms. They are words such as "is set" or "is the unfused" that look like the passive voice.
- The check counts many words as unapproved that are technical names in this project. Examples are `generate`, `reduce`, `implement`. The writers kept them when they name the thing.
- A project term can be less exact in STE. The writers recorded such cases. Example: the original uses "kernel" for two things. The STE version uses "user kernel" for the function of the application and "generated kernel" for the function that the runtime makes.
- When an original document changes, change the STE file too. The `--compare` command shows when the code or the numbers are different.
