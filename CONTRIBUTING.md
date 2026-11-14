# Contributing

Use Python 3.11+ and run commands from the repository root. For live inference,
follow the Conda setup in the README. For camera-free development:

```bash
python -m pip install -e '.[vision,dev]'
ruff check .
mypy
pytest -q
```

Tests use synthetic fixtures, mocks, and the real BoT-SORT backend. Keep ordinary tests independent of model
weights, network access, a physical camera, and GPU availability. Add focused tests
when changing counting, configuration validation, stream lifecycle, or persistence.

Include a short description of the problem, the resulting behavior, and the checks
run with each change. Document configuration or CLI changes in the README.

Keep datasets, weights, recordings, reports, credentials, and local run outputs
out of commits. Synthetic fixtures belong in `tests/fixtures/` with their origin
documented. Verify the publication rights for any new media or models.

Training and evaluation tools are described in
[documents/model-finetuning.md](documents/model-finetuning.md). Report validation
and held-out test results separately, and distinguish real-fruit experiments from
replica tests. Avoid inferring detector accuracy from total-count agreement.
