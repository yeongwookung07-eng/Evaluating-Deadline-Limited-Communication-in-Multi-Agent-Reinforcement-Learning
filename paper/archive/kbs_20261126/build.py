"""Build the main paper, supplement and portable LaTeX source archive."""

import json
from pathlib import Path
import re
import subprocess
import zipfile


PAPER = Path(__file__).resolve().parent
DOCUMENTS = ("kbs_main", "kbs_supplement")


def compile_document(name, output_log):
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]
    options = dict(cwd=PAPER, stdout=output_log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(latex, **options)
    aux = (PAPER / f"{name}.aux").read_text()
    if r"\bibdata{" in aux:
        subprocess.run(["bibtex", name], **options)
    elif r"\citation{" in aux and not (PAPER / f"{name}.bbl").exists():
        raise ValueError(f"{name} cites references without a bibliography")
    for _ in range(3):
        subprocess.run(latex, **options)

    log = (PAPER / f"{name}.log").read_text()
    for warning in ("undefined references", "undefined citations", "multiply defined",
                    "Missing character", "Author undefined", "Overfull", "Rerun to get cross-references right"):
        if warning in log:
            raise ValueError(f"{name}.log: {warning}")
    pdf_text = subprocess.check_output(
        ["pdftotext", "-layout", str(PAPER / f"{name}.pdf"), "-"], text=True
    )
    return pdf_text


def write_source_zip():
    inputs = [
        PAPER / filename for filename in (
            "cas-dc.cls", "cas-common.sty", "cas-model2-names.bst", "elsarticle-num.bst", "references.bib",
            "build.py", "plot_uniform_figures.py", "figure_inputs.json",
        )
    ]
    for name in DOCUMENTS:
        inputs.append(PAPER / f"{name}.tex")
        if r"\bibdata{" in (PAPER / f"{name}.aux").read_text():
            inputs.append(PAPER / f"{name}.bbl")
    inputs.extend(sorted((PAPER / "figures").glob("*.pdf")))
    with zipfile.ZipFile(PAPER / "kbs_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in inputs:
            archive.write(path, path.relative_to(PAPER))


def main():
    with (PAPER / "build.log").open("w") as output_log:
        main_text, supplement_text = [compile_document(name, output_log) for name in DOCUMENTS]
    state = dict(
        pages=sum(bool(page.strip()) for page in main_text.split("\f")),
        supplement_pages=sum(bool(page.strip()) for page in supplement_text.split("\f")),
        numerical_citations=bool(re.search(r"\[1[,\]]", main_text)),
        compiled=True,
    )
    (PAPER / "validation.json").write_text(json.dumps(state, indent=2) + "\n")
    write_source_zip()
    print(state)


if __name__ == "__main__":
    main()
