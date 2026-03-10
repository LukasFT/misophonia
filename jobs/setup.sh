#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install
uv sync --frozen --no-install-project
