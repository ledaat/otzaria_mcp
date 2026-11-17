# OtzariaPersonalVectors

Semantic search inside Otzaria for personal books (.txt) and the books of attached book
databases: embeds what Otzaria's vector set lacks with Otzaria's own model and publishes it into
the set as one extra segment (`.oxv`, store format 3).

- `OtzariaPersonalVectors.exe`: a window for anyone (`gui.py`, built by
  `.github/workflows/personal-vectors-exe.yml`): it checks what is missing on start;
  "התחל" embeds (on a GPU through DirectML when one works, else the CPU) and publishes after
  asking to close Otzaria; "עצור" stops (what was embedded is kept); "הסר מאוצריא" takes the
  segment out again.
- `otzaria_personal_vectors.py`: the same from a command line (`--check`, `--remove`, `--gpu`,
  `--help`). Needs `pip install numpy onnxruntime tokenizers zstandard`.
- `tests/synth_e2e.py`: end to end in a synthetic Otzaria folder with a stand-in model
  (`OXV_CHECK_SET` = otzaria-semantic-search's `examples/check_set` to have Otzaria's reader
  check the result).

Vectors are cached in `%LOCALAPPDATA%\otzaria-personal-vectors`, so a run can be stopped and
continued, and later runs embed only new or changed text. Unofficial: it writes into Otzaria's
vector folder in the format Otzaria reads today.
