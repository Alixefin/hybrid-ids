# Hybrid ML Threat Detection & Automated Response (Hybrid IDS)

This is a final-year Computer Science project. It is a five-layer intrusion detection and
response system: an **Isolation Forest** anomaly gate feeds a **Random Forest** classifier,
a policy engine maps each verdict to a response (alert, rate-limit, block, session kill,
quarantine), and a Flask dashboard shows the results. It is built around the CICIDS2017
dataset and can be deployed to PythonAnywhere's free tier.

> ⚠ **Synthetic data warning.** Until you swap in the real CICIDS2017 files, the models
> are trained on flows produced by `training/generate_synthetic_flows.py`. Every script
> prints a warning banner in that case, the dashboard footer shows a yellow **SYNTHETIC
> DATA** bar, and the metrics JSON carries a `"warning"` field. Those numbers only show
> that the pipeline works; **do not report them as evaluation results.**

---

## Architecture

```
 CICFlowMeter CSV ─► 1 Ingestion ─► pre-processing ─► 2 Isolation Forest ──score < θ──► Benign (counted, not stored)
 (upload / file)     engine/ingest   engine/preprocess   (benign-only)          │
                                                                          score ≥ θ
                                                                                ▼
                                                            3 Random Forest (class-weighted)
                                                            ŷ, p = share of trees voting ŷ
                                   ┌────────────────────────────┼─────────────────────────────┐
                           ŷ=Benign, p≥τ                   ŷ≠Benign, p≥τ                      p<τ
                        cleared false alarm              classified attack             "Unknown anomaly"
                                                                │                      (SOC alert only)
                                                                ▼
                               4 Policy engine (tiers, whitelist, 24 h repeat-offence doubling)
                                 └─► Enforcer: iptables / conntrack  [simulate | enforce]
                                 └─► Notifier: log / dashboard / optional SMTP
                                                                ▼
                    5 SQLite (8 tables) + Flask dashboard + audit_log
```

| Layer | Code |
|---|---|
| 1 Traffic ingestion | `engine/ingest.py` (CICIDS2017 raw headers, CICFlowMeter-V4 names, latin-1 files, 12-hour timestamps), `engine/preprocess.py` |
| 2 Isolation Forest | `engine/detector.py`: `score = -score_samples(x)`, θ = 99th percentile of benign **validation** scores (tunable at runtime) |
| 3 Random Forest | `engine/detector.py`: hard tree vote, p = share of trees, τ = 0.70; the decision rule is `decide()` |
| 4 Automated response | `engine/policy.py`, `engine/enforcer.py`, `engine/notifier.py`, orchestrated by `app/services.py` |
| 5 Logging & dashboard | `app/` (Flask factory, blueprints, templates), `app/models.py` (8 tables) |

### Pre-processing pipeline (`engine/preprocess.py`)
1. **Training rows:** drop NaN/±inf rows and duplicate flows, *before* the split so duplicates can't leak across it.
2. Identifier/time columns (`Flow ID`, IPs, ports, `Protocol`, `Timestamp`) are kept alongside each flow for logging and never used as features.
3. Drop near-constant features (one value in ≥ 99.95 % of rows).
4. Clip to the training [min, 99.9th percentile] range, then apply a signed `log1p` to heavy-tailed count/byte/time features.
5. Drop highly correlated features (|r| > 0.95, keeping the first in CICFlowMeter order).
6. Keep the top 30 features by Random Forest importance.
7. **Inference:** a flow can't simply be dropped (a zero-duration port-scan probe has `Flow Bytes/s = Infinity`), so non-finite values are *repaired* (±inf → clip bound, NaN → 0) and counted.

### Threat categories
| Category | CICIDS2017 labels |
|---|---|
| Benign | BENIGN |
| DoS/DDoS | DoS Hulk, DDoS, DoS GoldenEye, DoS slowloris, DoS Slowhttptest |
| Port Scan | PortScan |
| Brute Force | FTP-Patator, SSH-Patator |
| Web Attack | Web Attack – Brute Force / XSS / Sql Injection |
| Botnet (Malware) | Bot |
| Other (Infiltration/Heartbleed) | Infiltration, Heartbleed |
| Phishing-related | *reserved; CICIDS2017 has no phishing flows, so it is not trained* |

### Default response policy
One row per action. Rows that share a class and a minimum confidence form a **tier**, and
the highest tier with `min_confidence ≤ p` runs. Editable at *Admin → Policy & whitelist*.

| Threat | Severity | Confidence | Actions |
|---|---|---|---|
| DoS/DDoS | High | p ≥ 0.90 | temporary block 60 min + alert |
| DoS/DDoS | High | 0.70 ≤ p < 0.90 | rate-limit 30 min + alert |
| Brute Force | High | p ≥ 0.85 | terminate session + block 60 min + alert |
| Web Attack | High | p ≥ 0.85 | terminate session + block 30 min + alert |
| Botnet | Critical | p ≥ 0.85 | quarantine host + block controller 60 min + **urgent** alert |
| Port Scan | Medium | p ≥ 0.80 | rate-limit 15 min + alert |
| Other / Unknown | Medium-High | any (Unknown: p < τ) | alert only, **never** auto-blocked |

* The whitelist is checked first, by both the policy engine and the enforcer.
* Each earlier action of the same kind against the same address within 24 h doubles the duration, capped at 24 h.
* Actions expire through the expiry job.
* An action that is already active for the same target is suppressed rather than duplicated.

---

## Setup

Requires **Python 3.11** (64-bit). All pinned packages install from pre-built wheels.

```bash
cd hybrid_ids
python3.11 -m venv ../.venv
../.venv/bin/pip install -r requirements-dev.txt        # Windows: ..\.venv\Scripts\pip ...
```

(The examples below use `python` for the venv's interpreter.)

## Train, evaluate, run (synthetic data, works today)

```bash
python -m training.generate_synthetic_flows              # data/synthetic/synthetic_cicids2017.csv (121k flows)
python -m training.prepare_cicids2017 --source synthetic # data/prepared/flows.pkl + meta.json
python -m training.train_models                          # models_store/{if,rf}_<run>.joblib + run_<run>.json
python -m training.evaluate                              # reports/latest_metrics.json (protocols A, B, C)

flask --app run.py init-db                               # creates instance/hybrid_ids.db, seeds policy + whitelist, registers models
flask --app run.py create-admin
python run.py                                            # http://127.0.0.1:5000
```

Sign in and open **Offline evaluation**. Upload a CSV, or click **Run detection on this file**
next to `data/samples/demo_upload.csv` (6,060 labelled synthetic flows). If the file has a
`Label` column, the results page also scores the run against it (confusion matrix,
per-class P/R/F1/FPR).

Other CLI commands: `flask --app run.py run-detection <csv> [--dry-run]`,
`flask --app run.py expire-actions`, `flask --app run.py register-models`.

Useful options:
* `generate_synthetic_flows.py --rows 300000 --seed 7 --per-day`
* `train_models.py --theta-percentile 97 --tau 0.7 --n-features 30 --corr-threshold 0.95`
* `evaluate.py --protocols A,C --fast`

## Using the real CICIDS2017 dataset

1. Go to the Canadian Institute for Cybersecurity's official page:
   **https://www.unb.ca/cic/datasets/ids-2017.html**. At the bottom, follow the download
   link and fill in the short registration form. The CIC then gives you access to the
   download area.
2. Download **`GeneratedLabelledFlows.zip`**, the `TrafficLabelling` CSVs. This is the recommended
   version because it keeps the Flow ID, IPs, ports and timestamps, so alerts show real addresses and
   Protocol B can use real time. `MachineLearningCSV.zip` (`MachineLearningCVE`) also works: it has no
   identifier columns, so the day is taken from the file name and the row order stands in for time.
3. Unzip the **8 CSV files** straight into `hybrid_ids/data/real/`:
   ```
   data/real/Monday-WorkingHours.pcap_ISCX.csv
   data/real/Tuesday-WorkingHours.pcap_ISCX.csv
   data/real/Wednesday-workingHours.pcap_ISCX.csv
   data/real/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   data/real/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   data/real/Friday-WorkingHours-Morning.pcap_ISCX.csv
   data/real/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   data/real/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```
4. Swap the source. **This is the one-line change**; everything downstream is identical:
   ```bash
   python -m training.prepare_cicids2017 --source real
   python -m training.train_models
   python -m training.evaluate
   ```
   `--path D:/datasets/CICIDS2017/TrafficLabelling` reads the files from somewhere else.
   The real files' quirks are handled automatically:
   * leading spaces in headers and the duplicated `Fwd Header Length` column;
   * the cp1252 `–` in the web-attack labels;
   * about 288k blank rows in the Thursday file;
   * `Infinity`/`NaN` strings;
   * 12-hour timestamps without AM/PM.
5. Memory: all 2.8M flows need roughly 3–4 GB of RAM during preparation and training. On a smaller
   machine, use `prepare_cicids2017.py --source real --benign-fraction 0.3` (keeps every attack flow
   and 30 % of benign ones) and say so in your write-up.
6. Re-deploy: upload the new `models_store/` files, then use *Models & thresholds → Scan → Activate*.
   The synthetic-data banners disappear once the active models report `data_source = real`.

Known issues with the dataset itself (mislabelled flows, CICFlowMeter artefacts) are documented in
Engelen, Rimmer & Joosen, *"Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case
Study"* (IEEE SPW 2021). Cite it when you discuss your results.

---

## Evaluation (`training/evaluate.py`)

Each protocol retrains the full pipeline and reports **precision, recall, F1 and FPR**
(binary attack-vs-benign, per class, and macro-averaged) for **Isolation Forest alone**,
**Random Forest alone** and the **hybrid** system:

* **Protocol A:** stratified random 70/15/15.
  * Includes a θ sweep (95–99.5th percentile) and a τ sweep (0.5–0.9).
  * Includes an ablation without the correlation filter.
* **Protocol B:** temporal. The earliest 60 % of each capture day is used for train/val, the latest 40 % for testing.
  * Classes that only appear late in a day are listed under `classes_absent_from_train`.
* **Protocol C:** leave-one-attack-family-out. For the unseen family it reports the share that is:
  * missed by the IF gate;
  * cleared as benign by the RF;
  * raised as "Unknown anomaly";
  * labelled as another attack.

The results are written to `reports/metrics_<source>_<timestamp>.json` and
`reports/latest_metrics.json`, and shown under *Admin → Evaluation report*.

### What the synthetic smoke test already shows (pipeline behaviour, not results)
These come from synthetic data, so they illustrate the *mechanics* to look for once you
run on the real dataset:

1. **The IF gate caps hybrid recall.**
   * Hybrid recall can never exceed the gate's recall. At θ = p99, the IF passes most scans and
     slow DoS, but few low-and-slow application attacks (brute force, web, bot).
   * Lowering θ (e.g. p97) lets more of them through. The RF's "cleared false alarm" branch then
     absorbs most of the extra benign traffic.
   * The θ sweep in Protocol A quantifies this trade-off.
2. **The correlation filter hurts the IF.**
   * Removing redundant features (spec step 5) lowered IF recall from about 0.67 to 0.47 in
     Protocol A, while the RF was unaffected.
   * Tree ensembles implicitly weight a signal by how many columns carry it. See `ablation_no_corr_filter`.
3. **"Cleared false alarm" works against novelty detection.**
   * In Protocol C, the IF flags about 96 % of an unseen Port Scan family.
   * But an RF that never saw port scans votes *Benign* with p ≥ τ, so the hybrid clears
     them and detects about 1 %.
   * Unseen attacks surface as "Unknown anomaly" only when the RF is *unsure*.
   * This is a real limitation of the decision rule to discuss (possible fix: require
     p ≥ τ_benign > τ before clearing).
4. **Temporal drift within a category.** In Protocol B the RF saw Hulk-style DoS but not the
   late-Friday DDoS, and misses most of it even though both map to *DoS/DDoS*.

---

## Deployment

See **[DEPLOY.md](DEPLOY.md)** for the exact PythonAnywhere steps: "Manual configuration"
WSGI app, virtualenv, WSGI file pointing at `create_app()`, SQLite on persistent storage,
action expiry without a background scheduler, and the 30-day renewal. The hosted demo runs in **simulate mode only**; real
iptables/conntrack enforcement needs root on a separate lab VM.

## Security

* **Passwords:** hashed with werkzeug (scrypt).
* **Roles:** *analyst* can view data, change alert status and run offline evaluations. *admin*
  can also manage models, policies, the whitelist, users and manual responses, and read the audit log.
* **Lock-out:** 3 failed logins lock the account until an admin unlocks it. The lock state is
  derived from `audit_log`, so the `user` table stays exactly as specified and every
  lock/unlock is audited.
* **Web protections:**
  * CSRF tokens on every POST.
  * CSP / X-Frame-Options / nosniff headers; no inline scripts.
  * Session fixation protection and open-redirect protection.
  * Upload names sanitised, paths confined to the upload folder.
* **Enforcer:**
  * Validates every IP (single address only; loopback, multicast, reserved, link-local and unspecified addresses are refused).
  * Checks the whitelist before building any command.
  * Builds commands as **argument lists** and runs them with `subprocess.run(..., shell=False)` and a timeout.
  * Keeps its rules in its own `HYBRID_IDS` chain, tagged with comments.
* **Audit:** every automated and manual action goes to `audit_log`: logins, detection runs,
  each response action, expiries, releases, and policy/whitelist/model/user changes.

## Tests

```bash
python -m pytest            # 118 tests, ~40 s
```

| File | Covers |
|---|---|
| `tests/test_preprocess.py` | NaN/inf/duplicates, all-missing and constant columns, correlation filter, top-k, clipping, log transform, inference repair, missing columns, raw-header / V4-alias / latin-1 / timestamp normalisation, label mapping |
| `tests/test_detector.py` | Every branch and boundary of the decision rule, vectorised = scalar rule, hard tree-vote confidence, RF only runs on flows above θ, model mismatch, θ override |
| `tests/test_policy.py` | Tier selection per class and boundaries, fallback to alert-only, disabled rules, botnet targets, Unknown-never-blocks invariant, whitelist, repeat-offence doubling and cap |
| `tests/test_enforcer.py` | Simulate never executes, whitelist and invalid/dangerous IPs refused (including injection strings), exact command shapes (IPv4/IPv6), `shell=False` in enforce mode, failure handling |
| `tests/test_app.py` | Lock-out and unlock, RBAC, CSRF, open redirect, upload → detect → respond end to end, dry run, whitelisted attacker never blocked, expiry job, repeat-offence doubling, manual approval, CLI |

## Project structure

```
hybrid_ids/
├── app/                 Flask factory (__init__.py), models.py, services.py, security.py, cli.py,
│   ├── auth/            login / lock-out / password
│   ├── dashboard/       overview, alerts table + filters, threat detail
│   ├── responses/       response log, release, expiry job, manual approval
│   ├── admin/           models & thresholds, report, policy & whitelist, offline evaluation, users, audit
│   ├── templates/       Jinja2 + Bootstrap 5 + Chart.js
│   └── static/
├── engine/              ingest.py, preprocess.py, detector.py, policy.py, enforcer.py, notifier.py, schema.py
├── training/            generate_synthetic_flows.py, prepare_cicids2017.py, train_models.py, evaluate.py, pipeline.py
├── models_store/        trained .joblib files + run manifests (git-ignored)
├── data/                synthetic/, real/ (put CICIDS2017 here), prepared/, uploads/, samples/demo_upload.csv
├── reports/             evaluation metrics JSON
├── tests/               pytest
├── config.py            all settings (env-overridable)
├── run.py               dev server
└── pythonanywhere_wsgi.py
```

## Limitations

* The hosted demo processes uploaded CSVs rather than live traffic. PythonAnywhere offers no
  packet capture, and CICFlowMeter would need to run on the monitored network.
* The detector is only as good as CICIDS2017 is representative. Retrain before using it on another network.
* The synthetic generator was tuned to *resemble* CICIDS2017's structure. Its difficulty (how
  separable the classes are) is an assumption, not a measurement.
