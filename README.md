# Grading cigar wrapper leaf: an agent that measures, abstains and routes

Code and measured data behind the article *"An artificial intelligence agent to
grade, decide and route cigar wrapper tobacco leaf by type and location of
damage"*.

The system photographs a spread leaf, measures 69 physical features instead of
working on pixels, resolves the grade through three chained questions, abstains
when its confidence is low, and routes the leaf to a bin with a simulated KUKA
arm in RoboDK.

## Quick start

Python 3.12 and nothing else. One command runs the whole thing and tells you
whether what came out is what the article says:

    git clone https://github.com/EdgarSev17/cigar-leaf-grading-agent
    cd cigar-leaf-grading-agent
    pip install -r requirements.txt
    python run_all.py

About eight minutes on a laptop. No photographs to download, no graphics card,
no RoboDK, nothing to run in a particular order. `python run_all.py --quick`
does the same in ten seconds by skipping the fifteen-seed cross-validation.

It ends with this table, which is the point of the command:

    figure                                    article   this run
    ----------------------------------------------------------------------
    accuracy on the independent batch            84.8       84.8   matches
    Cohen's kappa                               0.705      0.705   matches
    at threshold 0.60, leaves decided            70.5       70.5   matches
    at threshold 0.60, accuracy on those         91.1       91.1   matches
    wrapper                                      91.5       91.5   matches
    XL left                                      68.2       68.2   matches
    XR right                                     78.9       78.9   matches
    error setting aside 10 %                     11.9       11.9   matches
    error setting aside 30 %                      8.9        8.9   matches

    Every figure matches the article.

It exits 0 if every figure matches and 1 if any of them moved. The robotic cell
is deliberately **not** part of this: verifying a paper should not require a
commercial licence. It is section 2 below, for whoever wants it.

## Key results

Measured on an independent batch of 112 Connecticut leaves, photographed on
another occasion, under different light, and never seen by the model. Every
figure is the mean of **six retrainings**, that is 672 decisions, which is the
protocol the article reports.

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

## What is in this repository

    run_all.py       the single command: runs everything, checks every figure
    code/            the pipeline: 26 modules, no absolute paths
    out/             the measured features the pipeline reads, and the model
    dataset/         labels, dubious cases and the human corrections
    results/         the independent batch, the article's figures, CIFRAS.md
    fotos_muestra/   nine leaves, three per grade, with their 3D meshes
    robodk/          the station, the control script, the signal protocol

Notable files:

- `code/reproduce.py` - reproduces Table III, Table IV and Figure 2 in one pass.
- `code/entrena.py` - trains variety and grade models from the feature tables.
- `code/clasifica.py` - the decision: three chained questions, the zero rule,
  the abstention threshold (`PISO_ARBOL = 0.60`).
- `code/cifras.py` - the cross-validated figures over the 627 training leaves.
- `out/celda/entorno1/logs/decisiones.jsonl` - one JSON line per decision the
  agent took in the cell: what it perceived, what it scored, what it decided and
  what it cost. The agent's behaviour can be checked against it rather than
  taken on trust.

## 1. Reproducing the reported figures

Needs Python 3.12. No photographs, no RoboDK, no graphics card.

**Step 1.** Get the repository and install the dependencies:

    git clone <this repository>
    cd <this repository>
    pip install -r requirements.txt

**Step 2.** Run everything with one command:

    python run_all.py

That is the whole thing, from the measured features to the results table: it
checks the environment, checks the data matches what the article declares,
trains and evaluates the article's protocol, recomputes the cross-validated
figures, grades one photograph end to end, and writes `results/RESULTS.md`.
About eight minutes on a laptop; `python run_all.py --quick` does the same in
ten seconds by skipping the fifteen-seed cross-validation, which is the slow
part.

**Step 3.** Read the last table it prints. This is the point of the command: it
does not only run, it compares every figure against what the article reports.

    figure                                    article   this run
    ----------------------------------------------------------------------
    accuracy on the independent batch            84.8       84.8   matches
    Cohen's kappa                               0.705      0.705   matches
    at threshold 0.60, leaves decided            70.5       70.5   matches
    at threshold 0.60, accuracy on those         91.1       91.1   matches
    wrapper                                      91.5       91.5   matches
    XL left                                      68.2       68.2   matches
    XR right                                     78.9       78.9   matches
    error setting aside 10 %                     11.9       11.9   matches
    error setting aside 30 %                      8.9        8.9   matches

`run_all.py` exits 0 if every figure matches and 1 if any of them does not, so
it can be dropped into a continuous integration job as it stands. A full run is
committed in `results/RESULTS.md`, so the output can be read without running
anything.

**What the single command runs.** Each step can also be run on its own:

| | |
|---|---|
| `python code/reproduce.py` | the article's protocol: six retrainings, 672 decisions |
| `python code/cifras.py` | the cross-validated figures over the 627 training leaves |
| `python code/clasifica.py <photo>` | one leaf, from image to grade to bin |
| `python code/entrena.py --familia logistica --sin-clase media_banda` | refits the model; `run_all.py --retrain` does this too |

Both options of the last one matter: without `--familia logistica` a margin of
0.2 points can flip the model family, and with it 39 points of accuracy.

`code/reproduce.py` prints, in this order:

- the accuracy and Cohen's kappa of the abstract, **84.8 %** and **0.705**;
- Table IV, how many leaves the system decides and how well it does on them, at
  the 0.60, 0.70 and 0.80 thresholds and setting aside the least confident 10 %
  and 20 %;
- Table III, the breakdown by grade;
- the four points of Figure 2.

**What you should see.** The same figures as the article, to the decimal,
because the seeds are fixed. If you change `--seeds` they will move a little;
that is the point of reporting a mean over six rather than a single fit.
Classifying the same batch with the single stored model in `out/modelo/` gives
86.6 % instead: both are correct, they measure different things, and the article
cites the mean.

## 2. Running the simulated cell

Optional. Nothing in the reported figures depends on it; the cell is here
because two claims in the article rest on it: that the ten generated programs
show no collisions, and that every decision was executed in simulation.

**RoboDK's free licence is enough.** Its camera is coarse, 3.05 x 6.02 mm per
pixel, ten times cruder than the model measures at, but the camera only raises
the event: the photograph is what gets graded.

**Step 1.** Install RoboDK and its Python API:

    pip install robodk

**Step 2.** Open the station `robodk/Entorno1_4clases.rdk` in RoboDK. It holds
the KUKA IONTEC KR 120 R2700, the belt, the nine bins and the ten pick-and-place
programs.

**Step 3.** If RoboDK is not installed in `C:/RoboDK`, say where it is:

    set ROBODK_DIR=D:/Programas/RoboDK        (Windows)
    export ROBODK_DIR=/opt/robodk             (Linux)

**Step 4.** Start the agent first, from the root of the repository. It loads the
model and waits:

    python code/agente_entorno1.py --n 9

**Step 5.** In a second terminal, from the same folder, start the cell:

    python robodk/Control_Senales.py

**Step 6.** Watch. The belt carries one leaf to the camera and stops; the cell
raises `HOJA_EN_CAMARA`; the agent grades the photograph and answers with the
destination bin as a four-bit code; the cell acknowledges and runs
`Recoger_Hoja` and `Dejar_Caja_NN`. The robot decides nothing and the agent
moves nothing.

**What you should see.** Run against a live station, the nine leaves come out
like this:

    1/9  wrapper   -> says wrapper   conf 1.000   bin 2   OK
    4/9  XL left   -> says wrapper   conf 0.745   code 9  deferred
    5/9  XL left   -> says XL left   conf 0.992   bin 3   OK
    9/9  XR right  -> says XR right  conf 0.936   bin 4   OK

Eight of the nine graded correctly, six routed to their bin, three deferred. The
one it got wrong was deferred rather than dropped in the wrong bin, which is the
abstention rule doing its job. Two it graded correctly were deferred anyway,
below the threshold: that is the cost of coverage, and the other face of the
same rule.

Nine leaves are nine leaves. They show the chain works end to end; they do not
measure accuracy. That comes from step 1, over all 112.

Every decision is appended to `out/celda/entorno1/registro_agente.csv` and to
`out/celda/entorno1/logs/decisiones.jsonl`.

## 3. Checking a measurement yourself

The nine leaves in `fotos_muestra/` are three wrapper, three XL left and three
XR right, all Connecticut, all from the independent batch, the same leaves the
reported figures are measured on. They are copied byte for byte, so they measure
exactly as they did.

    python code/clasifica.py fotos_muestra/capa/20260911_152039111_iOS.heic --ancho-cinta 43.8

    wrapper 43.8    XL left 49.5    XR right 48.9

Give it the tape width its folder was measured with, which is what
`--ancho-cinta` is for. What comes out is the row that leaf has in
`results/lote_independiente/rasgos_112.csv`, with areas agreeing to within
0.02 %. Without it, each photograph sets its own scale from its own tape and the
areas shift by about 10 %: that is the scale, not the measurement.

## The partitions are on disk, not regenerated

Fixing the seed already makes a split deterministic, but it leaves it depending
on the version of scikit-learn, on the order the rows happen to be in, and on
nobody ever touching the call. So the folds are written down instead, one JSON
file per partition under `results/particiones/`, holding **the leaf identifiers**
of each fold rather than row numbers: leaf ids survive a reordering of the tables
and a person can read them.

    {
      "nombre": "cifras_k5_sem0_0373e3ca",
      "n_splits": 5, "random_state": 0,
      "n_filas": 627, "n_grupos": 627,
      "huella_grupos": "0373e3caacdaf1ca",
      "folds": [{"test": ["hoja_0014", "hoja_0022", ...]}, ...]
    }

`code/particiones.py` is a drop-in for `StratifiedGroupKFold`: the first run
computes the folds and saves them, every run after that reads them back. Each
file records a fingerprint of the group list it was built for, so if the data
changes underneath, the run stops and says so instead of quietly producing
different numbers.

Verified: running the cross-validated figures with the folds computed, and then
again with them read from disk, gives **byte-identical output**.

## Why the photographs are not here

The image base belongs to a tobacco processing plant and is not published, and
neither is the name of the plant. What is published is everything derived from
the images: the 69 measured features per leaf, the labels given by the
technicians who set the plant standard, and the trained model. The nine sample
leaves are the exception, so that the measuring half can be run and checked.

The pipeline therefore splits in two:

- **photograph to features** (`rasgos_foto.py`, `segmentacion.py`, `zonas.py`,
  `agujeros.py`, `manchas.py`, `sudada.py`) is published and readable, and runs
  on the nine sample leaves, but cannot be re-run over the whole set.
- **features to grade to decision to routing** is fully reproducible, and it is
  the half the article's claims rest on.

## A note on language

The article and this README are in English. The code, its comments and the
column names of the data files are in Spanish, and they stay that way for a
reason: the grade names are the categories the trained model was fitted with,
and the column names are the keys the tables are joined on. Translating them
would mean retraining the model and rewriting every table, which would defeat
the purpose of publishing them. The terms you need:

| in the code and data | in the article |
|---|---|
| `capa` | wrapper |
| `banda` | binder |
| `media_banda` | half-binder, left out of the system |
| `xl_izq` | XL left |
| `xr_der` | XR right |
| `calidad` | grade |
| `variedad` | variety |
| `hoja` | leaf |
| `rasgos` | features |
| `cinta` | the reference tape that sets the scale |
| `caja`, `bandeja` | bin |
| `cola` | queue |

## Citation

    @inproceedings{sevilla2026wrapper,
      title  = {An artificial intelligence agent to grade, decide and route
                cigar wrapper tobacco leaf by type and location of damage},
      author = {Sevilla, Edgar and Torre, Luis and Puerto, Roberto and Loo, Luis},
      year   = {2026},
      note   = {Universidad Tecnologica de Honduras}
    }

## License

MIT, see `LICENSE`. The measured feature tables are published under the same
terms. The photographic base is not published.

## Contact

ejsevilla1@uth.hn
