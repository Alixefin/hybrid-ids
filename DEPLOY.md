# Deploying to PythonAnywhere (free tier)

The hosted demo runs as a **single-worker WSGI Flask app** in **simulate mode**. Detection
runs synchronously when someone uploads a CSV on the *Offline evaluation* page, or when
an admin clicks *Run detection on this file*. Nothing needs a background process.

> **Enforce mode is impossible on PythonAnywhere.** Real `iptables`/`conntrack`
> enforcement needs root and access to the host's netfilter / network namespace, which
> PythonAnywhere does not give you. There, the enforcer logs the exact command it
> *would* run (`status = simulated`). Use enforce mode only on a separate, self-hosted
> Linux lab VM (see the end of this file).

Replace `<username>` with your PythonAnywhere username everywhere below.

## 0. Before you start (on your own machine)

Train the models locally. The free tier's CPU allowance is too small to train on
PythonAnywhere, and you only need to upload the result.

```bash
python -m training.generate_synthetic_flows          # or put the real CSVs in data/real/
python -m training.prepare_cicids2017 --source synthetic
python -m training.train_models
python -m training.evaluate                          # optional: writes reports/latest_metrics.json
```

This creates `models_store/if_<run>.joblib`, `models_store/rf_<run>.joblib` and
`models_store/run_<run>.json` (about 10 MB on synthetic data).

**Use the same package versions on both machines.** `requirements.txt` pins
scikit-learn / numpy / scipy / joblib. A `.joblib` model only loads reliably with the
versions it was trained with.

## 1. Upload or clone the code

Open a **Bash console** (Consoles tab → Bash).

*Option A: Git* (the models and data are git-ignored, so you upload them in step 3):

```bash
cd ~
git clone https://github.com/<you>/<your-repo>.git hybrid_ids
```

*Option B: Files tab.* Zip the `hybrid_ids/` folder locally (leave out `.venv`, `data/synthetic`,
`data/prepared`, `instance/*.db`), upload the zip on the **Files** tab to `/home/<username>/`,
then:

```bash
cd ~ && unzip hybrid_ids.zip
```

Either way you should end up with `/home/<username>/hybrid_ids/run.py`.

## 2. Create a virtualenv and install the dependencies

```bash
mkvirtualenv --python=/usr/bin/python3.11 hybrid-ids
pip install --no-cache-dir -r ~/hybrid_ids/requirements.txt
```

* Inside a virtualenv, do **not** add `--user`. Use `pip install --user -r requirements.txt`
  only if you decide not to use a virtualenv, which is not recommended because the web app
  then shares packages with your whole account.
* `--no-cache-dir` matters on the free tier: 512 MB of disk. The pinned packages take
  **about 390 MB** installed (scipy 160 MB with its bundled libraries, pandas 75 MB,
  numpy 70 MB, scikit-learn 50 MB), which leaves roughly 100 MB for code, models, uploads
  and the database. Nothing is compiled: every pinned package has a `manylinux` wheel for
  Python 3.11.
* `requirements-dev.txt` (pytest) isn't needed on the server.
* Check your usage with `du -sh ~/.virtualenvs/hybrid-ids`, and see the *Account* page for the quota.
  If you run short, `rm -rf ~/.cache/pip`.

Later consoles: `workon hybrid-ids`.

**If the install runs out of quota**, reuse the scientific stack that PythonAnywhere
already ships with its Python 3.11 image instead of installing a second copy:

```bash
rmvirtualenv hybrid-ids
mkvirtualenv --python=/usr/bin/python3.11 --system-site-packages hybrid-ids
python -c "import numpy, pandas, scipy, sklearn, joblib; print(numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, joblib.__version__)"
pip install --no-cache-dir Flask==3.0.3 Flask-Login==0.6.3 Flask-SQLAlchemy==3.1.1 SQLAlchemy==2.0.36 APScheduler==3.10.4
```

Then, **on your own machine**, change the numpy / pandas / scipy / scikit-learn / joblib
pins in `requirements.txt` to the versions printed above, reinstall, and re-run
`train_models.py`, so the `.joblib` files match the server's scikit-learn.

## 3. Upload the trained models (and a demo CSV)

Files tab → go to `/home/<username>/hybrid_ids/models_store/` → upload the three files from
step 0 (`if_*.joblib`, `rf_*.joblib`, `run_*.json`). Optionally upload `reports/latest_metrics.json`
into `/home/<username>/hybrid_ids/reports/` so the *Evaluation report* page works.
`data/samples/demo_upload.csv` is already in the repository (6,060 synthetic flows) for demos.

## 4. Create the web app ("Manual configuration")

Web tab → **Add a new web app** → *Next* → choose **Manual configuration** (not the
"Flask" quick-start, which assumes a different layout) → **Python 3.11** → *Next*.

Then, on the web app's configuration page:

| Setting | Value |
|---|---|
| **Source code** | `/home/<username>/hybrid_ids` |
| **Working directory** | `/home/<username>/hybrid_ids` |
| **Virtualenv** | `/home/<username>/.virtualenvs/hybrid-ids` |
| **Static files**: URL `/static/` → Directory | `/home/<username>/hybrid_ids/app/static` |
| **Force HTTPS** | Enabled |

## 5. Point the WSGI file at the app factory

Web tab → click the **WSGI configuration file** link (`/var/www/<username>_pythonanywhere_com_wsgi.py`),
delete everything in it, and paste the contents of `pythonanywhere_wsgi.py` from this
repository. Then:

1. Replace `<username>`.
2. Nothing else. `SECRET_KEY` is generated automatically on first start and stored in
   `instance/secret_key` (git-ignored, mode 600). Set it in the WSGI file only if you want
   to manage it yourself.

In short, the file does this:

```python
import os, sys
PROJECT_HOME = "/home/<username>/hybrid_ids"
sys.path.insert(0, PROJECT_HOME); os.chdir(PROJECT_HOME)
os.environ.setdefault("ENFORCER_MODE", "simulate")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{PROJECT_HOME}/instance/hybrid_ids.db")
from app import create_app
application = create_app()          # PythonAnywhere looks for `application`
```

The SQLite database lives at `/home/<username>/hybrid_ids/instance/hybrid_ids.db`, on
PythonAnywhere's persistent storage. **Never put it under `/tmp`**, which is wiped.

## 6. Initialise the database and create an admin

In a Bash console:

```bash
workon hybrid-ids
cd ~/hybrid_ids
export DATABASE_URL="sqlite:////home/<username>/hybrid_ids/instance/hybrid_ids.db"
flask --app run.py init-db          # tables + default policy table + whitelist + registers models_store/
flask --app run.py create-admin     # prompts for username / email / password (min. 10 chars)
```

(`DATABASE_URL` above uses four slashes: `sqlite:///` plus the absolute path.)

## 7. Reload and test

Web tab → **Reload**. Open `https://<username>.pythonanywhere.com`, sign in, go to
**Offline evaluation**, and either upload `data/samples/demo_upload.csv` or click
**Run detection on this file** next to it. If anything fails, check the **error log**
and **server log** links on the Web tab.

## 8. Action expiry without a background scheduler

The free tier can't keep an APScheduler thread alive, so the scheduler is off
(`SCHEDULER_ENABLED=0`). Expired response actions are released:

* on demand: **Response log → Run expiry job now** (admin), and
* optionally once a day with the free tier's single **scheduled task** (Tasks tab):

  ```
  cd /home/<username>/hybrid_ids && DATABASE_URL=sqlite:////home/<username>/hybrid_ids/instance/hybrid_ids.db /home/<username>/.virtualenvs/hybrid-ids/bin/flask --app run.py expire-actions
  ```

In simulate mode nothing is ever actually blocked, so expiry only updates statuses.

## 9. Keep it alive: renew every month

Free web apps are disabled after about a month unless you extend them.
PythonAnywhere e-mails you before that happens. Log in and click **"Run until 1 month
from today"** on the Web tab. Put a reminder in your calendar. A lapsed app keeps its
files but stops serving until you renew it.

## 10. Free-tier limits to keep in mind

* **One web worker.** Uploads are scored inside the request, so keep them moderate. The app
  caps uploads at 50 MB / 200,000 rows (`MAX_UPLOAD_MB`, `MAX_UPLOAD_ROWS` in `config.py`).
  The 6,060-row demo file scores in a few seconds.
* **Outbound internet is allow-listed**, so SMTP alerts will probably not work. Alerts always
  appear on the dashboard, in the response log and in the audit log.
* **Disk quota: 512 MB.** Delete old uploads from the *Offline evaluation* page (✕ button).
* **Updating models:** train locally, upload the three new files, click
  *Models & thresholds → Scan models_store/*, then *Activate this run*.

## Self-hosted lab VM (enforce mode)

For real blocking, use an isolated Linux VM (Ubuntu 22.04+) you control:

```bash
sudo apt install python3.11-venv iptables conntrack
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export ENFORCER_MODE=enforce SCHEDULER_ENABLED=1 SECRET_KEY=...
sudo -E .venv/bin/flask --app run.py init-db
sudo -E .venv/bin/python run.py     # needs CAP_NET_ADMIN; bind to the lab network only
```

The enforcer creates its own `HYBRID_IDS` chain (jumped to from `INPUT` and `FORWARD`)
and tags every rule with an iptables comment. List the rules with
`sudo iptables -L HYBRID_IDS -n --line-numbers`, and flush them all with
`sudo iptables -F HYBRID_IDS`. Whitelist your management address **before** switching
to enforce mode.
