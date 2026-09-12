# Operations

```bash
icilval keys generate --out keys                      # validator signing key (keep keys/ private)
icilval store init store --key keys/validator.ed25519 --pool-id <pool_id>
icilval daemon --store store --pool pools/2026.09-v3 --key keys/validator.ed25519 \
  --queue queue --runs runs --admin-token "$ICIL_ADMIN_TOKEN" \
  --live https://<dashboard> --live-token "$ICIL_LIVE_TOKEN" \
  --mirror <owner>/icil-competition-results --docker-image icilval/model:dev
```

The daemon takes one entry from **each field's** queue in turn, runs the duel (each side in the
model container with `--network none`), publishes the signed record, mirrors the store and posts
live frames. `--once` runs a single pass. Without `--docker-image` the sides run in-process (the
BPP conda environment).

Round-robin rather than a worker per field: the store has one writer, and a lock fine enough to
let two fields publish at once would risk a torn index. The cost is that a long duel in one field
delays the other's queue. A field whose benchmark is not installed is logged and skipped, so an
absent plugin never stops another field and is never scored as empty.

### The queue is a directory

`--queue` names a **directory** holding one file per field (`queue/<track>.json`), because a
field's block counter advances with its own lineage. Upgrading from the single-file layout is one
move, which the validator refuses to guess:

```bash
mkdir -p queue && git mv queue.json queue/sensorimotor.json   # or plain mv
```

Pointing `--queue` at the old file raises and says exactly this, rather than starting from an
empty queue and silently losing the entries in it.

Smoke test end to end (in the BPP environment; `--model-dir` holds one directory per skill):

```bash
MUJOCO_GL=egl SDL_VIDEODRIVER=dummy icilval smoke --store /tmp/store --pool pools/smoke --model-dir ~/.cache/icilval/models/bpp-genesis --same-model
icilval store verify /tmp/store
```

Genesis: convert one public checkpoint per skill (`convert-ckpt --arch-name <skills.<skill>.architecture>`)
into `<dir>/<skill>`, publish the directory as that field's `baselines.<track>.repo`, pin the
revision, then `icilval genesis --king <repo>@<revision> …` (or simply queue it on an empty
throne). A field whose `baselines` entry is `null` opens with no king and crowns its first
entrant by genesis.

The drawing board runs pygame headless: `SDL_VIDEODRIVER=dummy` (the container sets it).

## Publishing

```bash
icilval pools push pools/2026.09-v3 --repo robotensor/icil-competition-pools   # then pin pools.pool_id
huggingface-cli upload robotensor/bpp-genesis models/genesis .                  # then pin baseline.revision
icilval store mirror store --repo robotensor/icil-competition-results --message "publish"
```

The daemon mirrors as it publishes; `store mirror` is for a store built by hand. Pass `--prune`
when the store was **rebuilt** rather than appended to — a schema change, a new pool — so the one
commit uploads every file and deletes every path the store no longer has. Without it the previous
layout's records and clips stay in the repo beside the new ones, under names nothing references.

## Container mode and filesystems

Model sides run as `docker run --network none --gpus all` with the model, pool, arch templates and
the side's run directory bind-mounted. Docker cannot bind-mount from FUSE filesystems (an encrypted
home such as gocryptfs fails with "change mount propagation … no such file or directory"), so on such
hosts keep `--runs`, `--pool`, `--arch` and any `--local-model` directories on a regular filesystem,
for example under `/var/lib/icilval`. The store and queue can live anywhere.
