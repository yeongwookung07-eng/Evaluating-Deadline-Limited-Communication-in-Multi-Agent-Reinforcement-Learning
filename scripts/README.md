# Submission packaging

`package_jaamas.py` builds the five JAAMAS documents, supplies complete editable LaTeX sources, standard upload filenames and SHA-256 checksums. Optional `--reviewer` writes a portable repository-based evidence ZIP under ignored `dist/`; it never submits to a journal or uploads externally. Existing reviewer archives are not overwritten. Scientific validation remains in `experiments/verify_results.py`.

The source ZIP includes the frozen Figure 1 plotting inputs and renderer. The reviewer ZIP includes the tested CPU constraints and table-reproduction helper. It preserves recorded scientific bytes and excludes regenerated working copies.
