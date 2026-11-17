"""Rolling replacement: old and new processes briefly share one VCF_INSTANCE_ID.

The handover contract under test:

- the new process's startup must not delete the old process's in-flight
  upload staging (the old upload still publishes);
- the new process can upload and serve indexed queries immediately;
- the old process's exit releases only its own occupancy — the new process's
  pinned queries keep reading their fixed version to the end;
- staging left behind by a process that really exited is still swept after
  the usual orphan grace period, and superseded versions are reclaimed once
  nobody uses them.
"""
import os
import time

import pysam
from conftest import SAMPLE, make_config, upload, vcf
from fastapi.testclient import TestClient

from app.main import create_app
from app.vcfbuild import GZ_NAME


def test_startup_of_new_process_keeps_old_process_in_flight_upload(tmp_path):
    """New process boots (same instance id) while the old one is mid-upload."""
    app_old = create_app(make_config(tmp_path, instance_id="vcf"))
    with TestClient(app_old) as c_old:
        # old process is mid-upload: a build dir with a partial file exists
        in_flight = os.path.join(app_old.state.staging_root, "build-123")
        os.makedirs(in_flight)
        marker = os.path.join(in_flight, "upload.vcf")
        with open(marker, "w") as fh:
            fh.write("partial upload still being written")

        # the replacement boots with the SAME VCF_INSTANCE_ID
        app_new = create_app(make_config(tmp_path, instance_id="vcf"))
        assert app_new.state.boot_id != app_old.state.boot_id
        with TestClient(app_new) as c_new:
            # startup did not touch the old process's staging
            assert os.path.isfile(marker)
            assert os.path.isdir(app_new.state.staging_root)

            # the old process finishes its own upload and publishes it
            r = upload(c_old, SAMPLE, base_version=0)
            assert r.status_code == 201 and r.json()["version"] == 1

            # the new process immediately uploads a replacement ...
            r = upload(c_new, vcf([(500, "A", "G")]), base_version=1)
            assert r.status_code == 201 and r.json()["version"] == 2
            # ... and immediately serves indexed queries
            q = c_new.get("/variants", params={"chrom": "chr1", "start": 500, "end": 500})
            assert q.status_code == 200
            assert q.json()["version"] == 2 and q.json()["count"] == 1

        # the new process's shutdown did not clean the old one's staging either
        assert os.path.isfile(marker)


def test_old_process_exit_keeps_new_process_pinned_query_alive(tmp_path):
    """Old process exits while the new one holds a pin on the old version."""
    app_old = create_app(make_config(tmp_path, instance_id="vcf"))
    app_new = create_app(make_config(tmp_path, instance_id="vcf"))
    store = app_new.state.store
    with TestClient(app_new) as c_new:
        with TestClient(app_old) as c_old:
            assert upload(c_old, SAMPLE, base_version=0).status_code == 201
            v1_path = store.version_info()["path"]

            # new process has an in-flight query pinning version 1
            pinned = store.pin("vcf", app_new.state.boot_id)
            assert pinned[0] == 1
            ref_id = pinned[3]

            # a replacement is published while that query is still open
            assert upload(c_old, vcf([(500, "A", "G")]), base_version=1).status_code == 201
        # <- old process exits here: unregister must release only its own rows

        # the new process is still registered as a live boot
        assert app_new.state.boot_id in store.live_boots(time.time(), 15.0)

        # reaper runs: the pinned old version must survive
        app_new.state.reap_once()
        v1_dir = os.path.join(app_new.state.cfg.data_dir, v1_path)
        assert os.path.isdir(v1_dir), "old process exit must not unpin the new process"

        # the pinned query still reads version 1 to the end, via its index
        tb = pysam.TabixFile(os.path.join(v1_dir, GZ_NAME))
        try:
            poss = [ln.split("\t")[1] for ln in tb.fetch("chr1", 99, 106)]
        finally:
            tb.close()
        assert poss == ["100", "105"]

        # fresh queries see version 2
        q = c_new.get("/variants", params={"chrom": "chr1", "start": 1, "end": 1000})
        assert q.json()["version"] == 2

        # query finishes -> the old version is reclaimed as usual
        store.unpin(ref_id)
        app_new.state.reap_once()
        assert not os.path.exists(v1_dir)
        assert store.current_version() == 2


def test_really_exited_process_staging_swept_after_grace(tmp_path):
    """Unregistered staging of a dead process: 60s grace, then reclaimed."""
    app_old = create_app(make_config(tmp_path, instance_id="vcf"))
    with TestClient(app_old) as c_old:
        assert upload(c_old, SAMPLE, base_version=0).status_code == 201
        leftover = os.path.join(app_old.state.staging_root, "half-build")
        os.makedirs(leftover)
        open(os.path.join(leftover, "upload.vcf"), "w").write("partial")
    # <- old process exited for real (unregistered); its staging is orphaned

    app_new = create_app(make_config(tmp_path, instance_id="vcf"))
    cfg = app_new.state.cfg
    with TestClient(app_new) as c_new:
        # inside the grace period the leftover is left alone
        app_new.state.reap_once()
        assert os.path.isdir(leftover)

        # past the grace period the reaper sweeps it (the reaper looks at the
        # per-boot staging dir's mtime)
        old = time.time() - 10 * cfg.orphan_grace
        os.utime(app_old.state.staging_root, (old, old))
        app_new.state.reap_once()
        assert not os.path.exists(leftover)

        # the new process's own staging survives its own reaper run
        assert os.path.isdir(app_new.state.staging_root)

        # published data kept serving the whole time
        q = c_new.get("/variants", params={"chrom": "chr1", "start": 100, "end": 103})
        assert q.status_code == 200
        assert [rec.split("\t")[1] for rec in q.json()["records"]] == ["100"]


def test_old_process_exit_releases_only_its_own_rows(tmp_path):
    """Direct check of unregister scoping under a shared instance id."""
    app_old = create_app(make_config(tmp_path, instance_id="vcf"))
    app_new = create_app(make_config(tmp_path, instance_id="vcf"))
    store = app_new.state.store
    with TestClient(app_new):
        with TestClient(app_old) as c_old:
            assert upload(c_old, SAMPLE, base_version=0).status_code == 201
            store.pin("vcf", app_old.state.boot_id)
            new_pin = store.pin("vcf", app_new.state.boot_id)
        # old process exited
        assert store.live_boots(time.time(), 15.0) == {app_new.state.boot_id}
        # old pin released, new pin untouched
        with store._txn() as con:
            rows = con.execute("SELECT boot_id FROM refs").fetchall()
            assert [r["boot_id"] for r in rows] == [app_new.state.boot_id]
        store.unpin(new_pin[3])
