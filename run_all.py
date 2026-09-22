# -*- coding: utf-8 -*-
"""ONE COMMAND THAT RUNS EVERYTHING.

    python run_all.py

From the published measured features to the results table, with no cells to run
in a particular order and nothing to remember. It takes about eight minutes on a
laptop and needs no graphics card and no RoboDK.

Most of that is step 4, the fifteen-seed cross-validation. `--quick` skips it and
still checks every figure the article reports, in about ten seconds.

What it does, in order:

    1. checks the environment against requirements.txt
    2. checks the data loads, matches what the article declares, and that
       the saved partitions are there
    3. trains and evaluates the article's protocol: six retrainings, each one
       measured on the 112 leaves of the independent batch
    4. recomputes the cross-validated figures over the 627 training leaves
    5. grades one photograph end to end, from image to bin
    6. writes results/RESULTS.md and compares every figure against the article

The last step is the point. It does not only run: it tells you whether what came
out is what the article says, figure by figure.

Options, none of them needed for a plain reproduction:

    --retrain     also refits and overwrites the stored model (adds ~2.5 min)
    --quick       skips step 4, the slowest; still checks every figure (10 s)
"""
import argparse
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PY = sys.executable

# What the article reports. Each entry: label, expected value, tolerance.
ESPERADO = [
    ("accuracy on the independent batch", 84.8, 0.1, "acc"),
    ("Cohen's kappa", 0.705, 0.002, "kappa"),
    ("at threshold 0.60, leaves decided", 70.5, 0.1, "dec060"),
    ("at threshold 0.60, accuracy on those", 91.1, 0.1, "acc060"),
    ("wrapper", 91.5, 0.1, "capa"),
    ("XL left", 68.2, 0.1, "xl"),
    ("XR right", 78.9, 0.1, "xr"),
    ("error setting aside 10 %", 11.9, 0.1, "e10"),
    ("error setting aside 30 %", 8.9, 0.1, "e30"),
]


def titulo(n, texto):
    print()
    print("=" * 74)
    print("  STEP %d  %s" % (n, texto))
    print("=" * 74)


def corre(cmd, etiqueta):
    """Runs a step, echoes it, and hands back its output."""
    t0 = time.time()
    p = subprocess.run([PY] + cmd, cwd=str(RAIZ), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    salida = (p.stdout or "") + (p.stderr or "")
    print(salida.rstrip())
    if p.returncode != 0:
        print()
        print("  !! %s failed with code %d" % (etiqueta, p.returncode))
        raise SystemExit(p.returncode)
    print("  [%s: %.1f s]" % (etiqueta, dt))
    return salida


def num(patron, texto, grupo=1):
    m = re.search(patron, texto)
    return float(m.group(grupo)) if m else None


def main():
    ap = argparse.ArgumentParser(description="Runs the whole system end to end.")
    ap.add_argument("--retrain", action="store_true",
                    help="also refit and overwrite the stored model")
    ap.add_argument("--quick", action="store_true",
                    help="skip the cross-validated figures, the slowest step")
    a = ap.parse_args()

    t_inicio = time.time()
    print()
    print("#" * 74)
    print("#  GRADING CIGAR WRAPPER LEAF - FULL REPRODUCTION")
    print("#  an agent that measures, abstains and routes")
    print("#" * 74)
    print("#  about %s on a laptop. No graphics card, no RoboDK needed."
          % ("10 seconds" if a.quick else "8 minutes"))
    print("#" * 74)

    # ---------------------------------------------------------------- 1
    titulo(1, "ENVIRONMENT")
    print("  Python   %s" % sys.version.split()[0])
    faltan = []
    for linea in io.open(RAIZ / "requirements.txt", encoding="utf-8"):
        linea = linea.strip()
        if not linea or linea.startswith("#") or "==" not in linea:
            continue
        paquete, version = linea.split("==")
        try:
            import importlib.metadata as md
            puesta = md.version(paquete.replace("-", "_").split("[")[0]
                                if paquete != "opencv-python-headless" else "opencv-python-headless")
            marca = "ok " if puesta == version else "!! "
            print("  %s%-26s wanted %-12s found %s" % (marca, paquete, version, puesta))
            if puesta != version:
                faltan.append(paquete)
        except Exception:
            print("  -- %-26s wanted %-12s NOT INSTALLED" % (paquete, version))
            if paquete not in ("torch", "torchvision", "robodk"):
                faltan.append(paquete)
    if faltan:
        print()
        print("  Versions differ from the ones everything was measured with.")
        print("  The figures may move slightly. pip install -r requirements.txt")

    # ---------------------------------------------------------------- 2
    titulo(2, "DATA")
    for f in ("out/manifiesto_limpio.csv", "out/clasificador_rasgos.csv",
              "out/agujeros.csv", "out/zonas.csv",
              "results/lote_independiente/rasgos_112.csv",
              "out/modelo/modelo.joblib"):
        p = RAIZ / f
        estado = "ok " if p.exists() else "!! MISSING"
        print("  %s %-46s %9s bytes" % (estado, f, p.stat().st_size if p.exists() else 0))
        if not p.exists():
            raise SystemExit("  A data file is missing; the repository is incomplete.")

    import csv
    lote = list(csv.DictReader(io.open(
        RAIZ / "results/lote_independiente/rasgos_112.csv", encoding="utf-8")))
    from collections import Counter
    rep = Counter(r["clase"] for r in lote)
    print()
    print("  independent batch: %d leaves   %s" % (len(lote), dict(rep)))
    print("  the article declares 112 leaves: 71 wrapper, 22 XL left, 19 XR right")
    if len(lote) != 112:
        raise SystemExit("  The batch does not have 112 leaves.")

    part = sorted((RAIZ / "results" / "particiones").glob("*.json"))
    if part:
        print("  saved partitions: %d files, read from disk, never regenerated"
              % len(part))
    else:
        print("  saved partitions: none yet; they will be computed and written")

    # ---------------------------------------------------------------- 3
    if a.retrain:
        titulo(3, "REFITTING THE STORED MODEL  (overwrites out/modelo/)")
        corre(["code/entrena.py", "--familia", "logistica",
               "--sin-clase", "media_banda"], "training")

    titulo(4 if a.retrain else 3, "THE ARTICLE'S FIGURES  (six retrainings, 672 decisions)")
    salida = corre(["code/reproduce.py"], "reproduction")

    medido = {
        "acc": num(r"accuracy, deciding every leaf\s+([\d.]+)", salida),
        "kappa": num(r"kappa ([\d.]+)", salida),
        "dec060": num(r"confidence threshold 0\.60\s+([\d.]+)", salida),
        "acc060": num(r"confidence threshold 0\.60\s+[\d.]+ %\s+([\d.]+)", salida),
        "capa": num(r"wrapper\s+\d+ of \d+\s+([\d.]+)", salida),
        "xl": num(r"XL left\s+\d+ of \d+\s+([\d.]+)", salida),
        "xr": num(r"XR right\s+\d+ of \d+\s+([\d.]+)", salida),
        "e10": num(r"sets aside 10 %\s+error\s+([\d.]+)", salida),
        "e30": num(r"sets aside 30 %\s+error\s+([\d.]+)", salida),
    }

    # ---------------------------------------------------------------- 4
    cv = ""
    if not a.quick:
        titulo(5 if a.retrain else 4, "CROSS-VALIDATED FIGURES  (627 training leaves)")
        cv = corre(["code/cifras.py"], "cross-validation")

    # ---------------------------------------------------------------- 5
    titulo(6 if a.retrain else 5, "ONE PHOTOGRAPH, END TO END")
    foto = "fotos_muestra/capa/20260911_152039111_iOS.heic"
    if (RAIZ / foto).exists():
        una = corre(["code/clasifica.py", foto, "--ancho-cinta", "43.8"], "one leaf")
    else:
        una = "  (the sample photographs are not in this copy)"
        print(una)

    # ---------------------------------------------------------------- 6
    titulo(7 if a.retrain else 6, "WHAT CAME OUT vs WHAT THE ARTICLE SAYS")
    lineas, todo_ok = [], True
    print()
    print("  %-38s %10s %10s   %s" % ("figure", "article", "this run", ""))
    print("  " + "-" * 70)
    for etiqueta, esperado, tol, clave in ESPERADO:
        v = medido.get(clave)
        if v is None:
            marca, ok = "could not read", False
        else:
            ok = abs(v - esperado) <= tol
            marca = "matches" if ok else "DIFFERS"
        todo_ok = todo_ok and ok
        print("  %-38s %10s %10s   %s"
              % (etiqueta, esperado, "-" if v is None else v, marca))
        lineas.append((etiqueta, esperado, v, ok))

    dt = time.time() - t_inicio
    print()
    print("  " + ("Every figure matches the article." if todo_ok
                  else "Some figures differ. See the table above."))
    print("  total time: %.0f s" % dt)

    # ------------------------------------------------------- results table
    md = ["# Results of a full run", "",
          "Produced by `python run_all.py` on %s."
          % time.strftime("%Y-%m-%d %H:%M"),
          "", "Python %s, %.0f s." % (sys.version.split()[0], dt), "",
          "## The article's figures, checked", "",
          "| figure | article | this run | |", "|---|---|---|---|"]
    for etiqueta, esperado, v, ok in lineas:
        md.append("| %s | %s | %s | %s |"
                  % (etiqueta, esperado, "-" if v is None else v,
                     "matches" if ok else "DIFFERS"))
    md += ["", "## Full output", "", "### The article's protocol", "",
           "```", salida.rstrip(), "```", ""]
    if cv:
        md += ["### Cross-validated figures", "", "```", cv.rstrip(), "```", ""]
    md += ["### One photograph, end to end", "", "```", una.rstrip(), "```", ""]

    destino = RAIZ / "results" / "RESULTS.md"
    io.open(destino, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("  written to results/RESULTS.md")
    print()
    raise SystemExit(0 if todo_ok else 1)


if __name__ == "__main__":
    main()
