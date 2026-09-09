# Operations

```bash
icilval keys generate --out keys                      # validator signing key (keep keys/ private)
icilval store init store --key keys/validator.ed25519 --pool-id <pool_id>   # the catalogue's id
icilval daemon --store store --pool catalogues/2026.09-v4 --key keys/validator.ed25519 \
  --queue queue/queue.json --runs runs --admin-token "$ICIL_ADMIN_TOKEN" \
  --live https://<dashboard> --live-token "$ICIL_LIVE_TOKEN" \
  --mirror <owner>/icil-competition-results --docker-image icilval/model:dev
```

The daemon pops the queue, runs the duel (each side in the model container with `--network none`),
publishes the signed record, mirrors the store and posts live frames. `--once` runs a single entry.
Without `--docker-image` the sides run in-process (the BPP conda environment).

Smoke test end to end (in the BPP environment; `--model-dir` holds one directory per skill):

```bash
MUJOCO_GL=egl SDL_VIDEODRIVER=dummy icilval smoke --store /tmp/store --pool catalogues/smoke --model-dir ~/.cache/icilval/models/bpp-genesis --same-model
icilval store verify /tmp/store
```

Genesis: convert one public checkpoint per skill (`convert-ckpt --arch-name <skills.<skill>.architecture>`)
into `<dir>/<skill>`, publish the directory as `spec.baseline.repo`, pin the revision, then
`icilval genesis --king <repo>@<revision> …`
(or simply queue it on an empty throne).

The drawing board runs pygame headless: `SDL_VIDEODRIVER=dummy` (the container sets it).

## Publishing

```bash
icilval catalogue push catalogues/2026.09-v4 --repo robotensor/icil-competition-pools   # then pin catalogue.pool_id
huggingface-cli upload robotensor/bpp-genesis models/genesis .                  # then pin baseline.revision
icilval store mirror store --repo robotensor/icil-competition-results --message "publish"
```

The daemon mirrors as it publishes; `store mirror` is for a store built by hand. Pass `--prune`
when the store was **rebuilt** rather than appended to — a schema change, a new catalogue — so the one
commit uploads every file and deletes every path the store no longer has. Without it the previous
layout's records and clips stay in the repo beside the new ones, under names nothing references.

## Container mode and filesystems

Model sides run as `docker run --network none --gpus all` with the model, catalogue, arch templates and
the side's run directory bind-mounted. Docker cannot bind-mount from FUSE filesystems (an encrypted
home such as gocryptfs fails with "change mount propagation … no such file or directory"), so on such
hosts keep `--runs`, `--pool`, `--arch` and any `--local-model` directories on a regular filesystem,
for example under `/var/lib/icilval`. The store and queue can live anywhere.

## Prompt generation

A duel generates every unit's prompt before either side runs (`materializing`): worker processes
(`--workers`, default `generation.workers`) run the simulators' generators under the validator's
LIBERO config, reading the grasp-source files from the raw cache (`--raw`, default
`~/.cache/icilval/raw`; `icilval catalogue build --fetch` records their hashes, the first duel
fetches them) and the vendored BPP checkout (`--bpp-root`, default `vendor/behavior_prompting`).
Prompts land in `<runs>/<event>/assets/<unit_id>.npz`; in-process sides read them there, container
sides get the directory mounted read-only at `/assets` (`run-side --assets`). Every prompt is
published by hash (`media/<sha[:2]>/<sha>.npz`) and `store verify` checks it is there. A duel
whose materializing exceeds `budgets.materialize_wall_seconds` fails; a unit whose prompt could
not be generated after `generation.max_attempts` and three substitutions runs void.
