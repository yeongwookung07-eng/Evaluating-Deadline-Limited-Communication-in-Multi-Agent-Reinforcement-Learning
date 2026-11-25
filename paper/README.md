# Paper and supplement

- [Main paper](kbs_main.pdf): 15 pages, including references.
- [Supplement](kbs_supplement.pdf): 9 pages.
- [LaTeX source archive](kbs_source.zip).

Source files, bibliography, figures and style files are included here. With the required LaTeX tools installed, run `python build.py` from this directory to build both PDFs. Historical filenames are retained.

The paper analyzes delivery, computation and control in deadline-limited MARL communication. Experiment code, recorded outputs and checkpoints are in sibling directories; [the experiment map](../docs/EXPERIMENTS.md) and [findings](../docs/RESULTS.md) provide a Markdown guide.

The November 25 edition includes 29 references in citation order, both author emails, the postal address, a 222-word abstract, aligned email/equation layout, CRediT and separate data/code availability. LaTeX requires `xurl` for long URLs. No new scientific measurements are introduced.

Typography update (2026-11-25): non-heading text and figure labels use 10pt throughout both PDFs. Mathematical scripts retain standard sizing. All scientific prose, 18 tables and plotted numerical data were verified unchanged. See `typography_validation_20261125.json`. To redraw the figures from the bundled frozen inputs, run `python plot_uniform_figures.py` (NumPy/Matplotlib), then `python build.py`. No training is needed.
