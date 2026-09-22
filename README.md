# Grading cigar wrapper leaf: an agent that measures, abstains and routes

Code and measured data behind the article *"An artificial intelligence agent to
grade, decide and route cigar wrapper tobacco leaf by type and location of
damage"*.

The system photographs a spread leaf, measures 69 physical features instead of
working on pixels, resolves the grade through three chained questions, abstains
when its confidence is low, and routes the leaf to a bin with a simulated KUKA
arm in RoboDK.

## Key results

Measured on an independent batch of 112 Connecticut leaves, photographed on
another occasion, under different light, and never seen by the model. Every
figure below is the mean of **six retrainings**, that is 672 decisions, which is
the protocol the article reports.

| | |
|---|---|
| Accuracy, deciding every leaf | **84.8 %**, Cohen's kappa 0.705 |
| Most frequent grade (floor) | 63.4 % |
| At the 0.60 confidence threshold | decides **70.5 %** of the leaves, gets **91.1 %** of those right |
| By grade | wrapper 91.5 %, XR right 78.9 %, XL left 68.2 % |
| Variety, Habano vs Connecticut | 111 of 112 and 217 of 217 outside training |
| Measured features vs pixels | 77.9 % against 40.8 % on the independent batch |
| Cost of the confident model | gradient boosted trees let 17 misgraded leaves through above 0.90 confidence, logistic regression 7 |
| RoboDK cell | 9 bins, ten generated programs, no collisions |

One command reproduces all of it, without photographs. See **Reproducing**.

## What is in this repository

    code/       the pipeline: 26 modules, no absolute paths
    out/        the measured features the pipeline reads, and the trained model
    dataset/    labels, dubious cases and the human corrections
    results/    the independent batch, the article's figures, CIFRAS.md
    fotos_muestra/  nine leaves, three per grade, to run the measuring half
    robodk/     the simulated cell: the .rdk station, protocol and builder

Notable files:

- `code/reproduce.py` — reproduces Table III, Table IV and Figure 2 of the
  article in one pass, from measured features, without photographs.
- `code/entrena.py` — trains variety and grade models from the feature tables.
- `code/clasifica.py` — the decision itself: three chained questions, the zero
  rule, the abstention threshold (`PISO_ARBOL = 0.60`).
- `code/cifras.py` — the cross-validated figures over the 627 training leaves.
- `out/celda/entorno1/logs/decisiones.jsonl` — one JSON line per decision the
  agent took in the cell: what it perceived, what it scored, what it decided and
  what it cost. This is the audit trail; the agent's behaviour can be checked
  against it rather than taken on trust.
- `robodk/LEEME_Entorno1.md` — the cell: 9 bins on a 1600 mm arc, the 4-bit
  signal protocol, and which file owns each number.

## Why the photographs are not here

The image base belongs to a tobacco processing plant and is not published, and
neither is the name of the plant. What is published is everything derived from
the images: the 69 measured features per leaf, the labels given by the
technicians who set the plant standard, and the trained model.

Nine leaves are the exception, in `fotos_muestra/`: three of each grade the
independent batch contains, copied byte for byte so that they measure exactly as
they did. They are there so the measuring half can be run and checked, not for
training. To measure one, give it the tape width its folder was measured with,
which is what `--ancho-cinta` is for:

    python code/clasifica.py fotos_muestra/capa/20260911_152039111_iOS.heic --ancho-cinta 43.8

    capa 43.8    xl_izq 49.5    xr_der 48.9

The values that come out are the row that leaf has in
`results/lote_independiente/rasgos_112.csv`: areas agree to within 0.02 %, the
rest is the rounding of the published decimals. Without `--ancho-cinta` each
photograph sets its own scale from its own tape and the areas shift by about
10 %, which is the scale, not the measurement.

The batch contains no binder leaf --- the article says so among its limitations ---
so neither does the sample.

The pipeline therefore splits in two:

- **photograph → features** (`rasgos_foto.py`, `segmentacion.py`, `zonas.py`,
  `agujeros.py`, `manchas.py`, `sudada.py`) is published and readable, but
  cannot be run here, because it needs the images.
- **features → grade → decision → routing** is fully reproducible, and it is
  the half the article's claims rest on.

## Reproducing

Python 3.12. From the root of the repository:

    pip install -r requirements.txt
    python code/reproduce.py

Takes under a minute on a laptop and needs no graphics card. It prints the
accuracy and kappa of the abstract, Table III (per grade), Table IV (how many
leaves the system decides and how well it does on them) and the four points of
Figure 2.

**The numbers will be close but need not be identical**, and the article is
written that way on purpose: partitions are drawn at random, so what is reported
is the mean over six retrainings and, elsewhere, the range over fifteen. A
difference smaller than that range means nothing. On the machine used for the
article the command above prints 84.8 % exactly.

Two other entry points:

    python code/entrena.py --familia logistica --sin-clase media_banda
    python code/cifras.py

Both options of the first matter: without `--familia logistica` a margin of 0.2
points can flip the model family, and with it 39 points of accuracy;
`--sin-clase media_banda` because the final model leaves that grade out.

Note that `reproduce.py` retrains from the feature tables, while the model
stored in `out/modelo/` is a single fit. Classifying this batch with that one
stored model gives 86.6 %; the 84.8 % of the article is the mean of six, which
is the defensible number and the one to cite.

### The robotic cell

The station is `robodk/Entorno1_4clases.rdk`: the nine-bin version, one bin per
grade and variety plus one for review. The code's docstrings call it *Entorno1*,
which is the same cell before the bins were renumbered from eleven to nine.

Needs RoboDK installed. If it is not in `C:/RoboDK`, set `ROBODK_DIR` first:

    set ROBODK_DIR=D:/Programas/RoboDK
    python code/agente_entorno1.py --n 10

The agent sends the destination bin as a 4-bit code and waits for
acknowledgement; the robot decides nothing and the agent moves nothing. The
protocol is in `robodk/LEEME_Entorno1.md`.

One caveat, because it will bite on another machine: a script embedded inside
the station writes the leaf queue to an absolute path
(`D:/tesis-tabaco/out/celda/entorno1/cola_hojas.csv`). Open the station in
RoboDK, look under `Scripts`, and point it at your own copy. The rest of the
repository has no absolute paths; this one lives inside the `.rdk` and cannot be
resolved from outside it.

## Citation

    @inproceedings{sevilla2026wrapper,
      title  = {An artificial intelligence agent to grade, decide and route
                cigar wrapper tobacco leaf by type and location of damage},
      author = {Sevilla, Edgar and Torre, Luis and Puerto, Roberto and Loo, Luis},
      year   = {2026},
      note   = {Universidad Tecnol\'ogica de Honduras}
    }

## License

MIT, see `LICENSE`. The measured feature tables are published under the same
terms. The photographic base is not published.

## Contact

ejsevilla1@uth.hn
