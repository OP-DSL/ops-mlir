# How the STE documents are written

The files in this directory are the Simplified Technical English (ASD-STE100) version of the
documents in `docs/` and of the project `README.md`. Each file here has the same name as its
original and the same sections, the same code blocks and the same numbers.

These files follow the ASD-STE100 writing rules, but they are not certified. The rules are
applied with some freedom. Where a rule makes a sentence unclear, clarity has priority. The
program `ste_check.py` in this directory tests the rules that a program can test.

## 1. Rules that the writers use

| # | rule | bad | good |
|---|---|---|---|
| 1 | One sentence has 25 words or less. An instruction has 20 words or less. | (a long sentence with a semicolon and two clauses) | Two short sentences. |
| 2 | One paragraph has 6 sentences or less and one topic. | | |
| 3 | Use the active voice. Use the passive voice only if the actor is not known or not important. | The loop is queued by the runtime. | The runtime puts the loop in the queue. |
| 4 | Use the simple tenses: present, past, future. Do not use the perfect or the progressive. | The queue has been flushed. | The runtime flushed the queue. |
| 5 | Do not use the -ing form as a verb or as a noun. Use the infinitive or a new sentence. | Fusing loops saves memory traffic. | When the planner fuses loops, the kernel makes less memory traffic. |
| 6 | Use "can" for ability and "must" for an obligation. Do not use "may", "should", "might". | The loop may run on the host. | The loop can run on the host. |
| 7 | Use "because", "after", "when". Do not use "since", "once", "while". | Once the queue is full, it flushes. | When the queue is full, it flushes. |
| 8 | Use a noun cluster of 3 words or less. | the fused group stencil access offset list | the offsets of the stencil access in the group |
| 9 | Do not omit the articles "the" and "a". | Planner moves loop. | The planner moves the loop. |
| 10 | Do not use semicolons, contractions or Latin abbreviations (e.g., i.e., etc.). Use a list. | | |
| 11 | Write an instruction in the imperative. Write one instruction in one sentence. Use a numbered list for steps. | | Run the program. Read the log. |
| 12 | Use one word for one meaning. Do not use synonyms to vary the text. | queue, list, pipeline for one thing | queue |
| 13 | Use approved words. Use a technical name when no approved word exists (section 2). | utilize, perform, ensure, obtain | use, do, make sure, get |
| 14 | Do not use idioms, metaphors, jokes or phrasal verbs with many meanings. | the runtime picks up the slack | the runtime does the work |
| 15 | Write numbers as numerals. Put a space before a unit. | two seconds, 3s | 2 s |
| 16 | Start a paragraph with its main point. | | |

Use the words below in place of common unapproved words.

| do not use | use |
|---|---|
| utilize | use |
| perform | do |
| ensure | make sure |
| obtain | get |
| provide | give |
| execute (a loop or a kernel) | run |
| require | need |
| however | but (or start a new sentence) |
| therefore, thus, hence | so, or "because of this" |
| prior to | before |
| approximately | about |
| additional | more |
| multiple, various, numerous | many, or "different" |
| via | with, through |
| currently | now |
| typically | usually |

## 2. What stays the same as in the original

- All code blocks and all IR listings: copy them character by character. Do not translate them.
- Identifiers in `code spans`: environment variables, function names, file names, command lines.
- Numbers, units, table values and the results of measurements.
- The number and the order of the headings, tables and code blocks.
- Figures: keep the image links. Change the path if the file location changes (the images are in `../img/`).

Change these things:

- Rewrite all prose and table text in STE.
- Change a relative link to a file in `docs/` only when the target is an STE file. The STE file has the same name, so the link text can stay the same.
- Change a relative link that leaves the `docs/` directory: add `../` (for example `data/x.json` becomes `../data/x.json`).
- Add this line below the title of each file:
  `> This is the Simplified Technical English (ASD-STE100) version of [NAME.md](../NAME.md). Code, listings and numbers are the same as in the original.`

## 3. Project vocabulary: one word, one meaning

Use these words and no others for these things. They are technical names. STE allows a technical
name when the project needs it, and the writer must use it in the same way everywhere.

| word | meaning | do not use for it |
|---|---|---|
| loop | one `ops_par_loop` call in the application | |
| user kernel | the C++ function that the application gives to a loop | kernel (alone), body, callback |
| generated kernel | the function that the runtime makes for one group of loops | fused kernel, launch unit |
| GPU kernel | the function that runs on the GPU (`gpu.func`) | |
| group | the loops that share one generated kernel | cluster, bucket, set |
| fuse | to put loops in the same group | merge, combine |
| queue | the list of loops that did not run yet | list, pipeline, buffer |
| flush | to run all loops in the queue (also the noun) | drain, commit |
| segment | the loops between two stock loops in one flush | stretch, chunk |
| plan | the groups that the planner chooses for a segment | schedule |
| planner | the part of the runtime that makes the plan | |
| dat | an OPS data set (`ops_dat`) | array, field (field means the IR type) |
| field | the type of a dat in the stencil IR | |
| buffer | a region of memory | |
| range | the index range of a loop | bounds, extent |
| box | the bounding box of the ranges of a group | |
| stencil point | one offset of a stencil | tap, neighbour |
| guard | the test that limits a member of a group to its own range | mask, predicate |
| member | one loop in a group | |
| stock loop | a loop that runs with the stock OPS code on the host | host loop, fallback loop |
| fallback | the closure that runs a loop as a stock loop | |
| translate | to change a user kernel from C++ to MLIR | convert, transpile |
| lower | to change IR to a lower level IR | |
| compile | to change LLVM IR to machine code | |
| run | to do the work of a loop or a kernel | execute |
| launch | to start a generated kernel (GPU or engine call) | |
| reduction | an OPS argument that gives one value from many points | |
| contribution | the value of one point for a reduction | |
| scratch buffer | the buffer that holds the contributions | temporary array |
| fold | to change a scratch buffer to one value | collapse, reduce (verb) |
| cache hit, cache miss | the key is in the cache, the key is not in the cache | |
| key | the digest that identifies a queue shape (`ModuleKey`) | hash |
| backend | `seq`, `openmp` or `cuda` | target, platform |
| host | the CPU and its memory | |
| device | the GPU and its memory | |
| QA | the CloverLeaf check of the result | validation |

If a word of the original text is not in this table and is not in the approved dictionary, use a
simple word that has the same meaning. If the meaning needs a technical word, use the word of the
original and use it in the same way everywhere.

## 4. Tests

```bash
python3 docs/ste/ste_check.py docs/ste/FILE.md                 # style report
python3 docs/ste/ste_check.py --summary docs/ste/*.md          # counts only
python3 docs/ste/ste_check.py --compare docs/FILE.md docs/ste/FILE.md   # same code, numbers, headings
```

The style report lists these items: sentences that are too long, paragraphs with more than 6
sentences, semicolons, contractions, Latin abbreviations, possible passive and perfect forms,
unapproved words, and the words "may", "should", "since", "once" and "while". The hints
`soft-word` and `ing-word` are low priority. They often show a technical name that is correct.

The check cannot find a wrong meaning. A person must read the text again.
