# Cloud Offload Readiness V1

## What this batch adds

The project now has a headless, container-ready NFL research worker. It is separate from the desktop GUI and makes no cloud change by itself.

```bash
docker compose -f docker-compose.nfl-research.yml build
docker compose -f docker-compose.nfl-research.yml run --rm nfl-research --dry-run
```

There is also a headless CFB portability check. It mounts the existing local
data and model folders rather than copying them into an image, and defaults to
the canonical CFB `--dry-run` mode:

```bash
docker compose -f docker-compose.cfb-runner.yml build
docker compose -f docker-compose.cfb-runner.yml run --rm cfb-runner --dry-run
```

The compose file mounts `./data/nfl` at runtime and reads API keys from the existing `.env`; neither data nor credentials are baked into the image.

## What this batch does not do

- It does not create an AWS, cloud, Docker registry, or billing resource.
- It does not move the 1.9 GB local CFB data set.
- It does not containerize the PySide desktop GUI.
- It does not change the frozen CFB model, CFB history, or canonical game-day workflow.

## Migration sequence

1. Run the NFL worker locally in Docker and compare its source ledger with the native run.
2. Run the CFB headless `--dry-run` container against the existing local data/model mounts without altering its research behavior.
3. Deploy the headless runner to a private staging environment with persistent storage and managed secrets.
4. Run the cloud copy in parallel with the local CFB workflow before treating it as authoritative.
5. Only then move scheduled collection jobs off the Mac.

The current NFL SQLite file is intentionally single-worker local storage. A future multi-worker cloud deployment will need an explicitly planned persistent database/storage migration rather than a silent SQLite lift-and-shift.
