"""
Download Texas Railroad Commission datasets for the TROA RAG project.

Pulls:
  - PDF user manuals (the primary RAG corpus)
  - A sample of structured data files (field rules, inspections, etc.)
  - A sample of imaged W-1 drilling permits (optional, large)

Usage:
    python data/download_data.py                 # Just manuals (fast, ~50 MB)
    python data/download_data.py --include-data  # Also structured data (~500 MB)
    python data/download_data.py --include-all   # Everything including imaged PDFs (~5 GB)

Files land under data/raw/ in subdirectories by category.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


RAW_DIR = Path(__file__).parent / "raw"
USER_AGENT = "TROA-RAG-Research/0.1 (educational; contact: github.com/drealchux)"


@dataclass
class Dataset:
    """One downloadable dataset entry from the RRC page."""

    name: str
    category: str            # manual, structured, imaged
    url: str
    filename: str
    description: str
    sha256: Optional[str] = None  # filled in after first verified download


# Curated list of the most valuable PDF manuals for the RAG corpus.
# Each manual describes one form or dataset, with field-level layouts and procedural rules.
MANUALS: list[Dataset] = [
    Dataset(
        name="drilling_permit_master",
        category="manual",
        url="https://www.rrc.texas.gov/media/ezxjqdmn/oga049.pdf",
        filename="oga049_drilling_permit_master.pdf",
        description="Drilling Permit Master data layout and rules (since 1976)",
    ),
    Dataset(
        name="drilling_permit_master_latlong",
        category="manual",
        url="https://www.rrc.texas.gov/media/zgccdjpi/drillingpermitmasterpluslatitudeslongitudes_oga049m_july1.pdf",
        filename="oga049m_drilling_permit_master_latlong.pdf",
        description="Drilling Permit Master with latitudes/longitudes",
    ),
    Dataset(
        name="pending_drilling_permits",
        category="manual",
        url="https://www.rrc.texas.gov/media/pkgkvkcb/pendingdrillingpermits.pdf",
        filename="pending_drilling_permits.pdf",
        description="Drilling permits pending approval",
    ),
    Dataset(
        name="horizontal_drilling_permits",
        category="manual",
        url="https://www.rrc.texas.gov/media/yi0p3f3j/horizontaldrillingpermits.pdf",
        filename="horizontal_drilling_permits.pdf",
        description="Horizontal drilling permits dataset",
    ),
    Dataset(
        name="oil_gas_annual_field",
        category="manual",
        url="https://www.rrc.texas.gov/media/f1sekd2u/oga100k.pdf",
        filename="oga100k_oil_gas_annual_field.pdf",
        description="Oil/Gas Annual Report Field Table",
    ),
    Dataset(
        name="oil_gas_field_name_numbers",
        category="manual",
        url="https://www.rrc.texas.gov/media/wmpp4lmh/oga040.pdf",
        filename="oga040_field_name_numbers.pdf",
        description="Oil & Gas field name and number dataset",
    ),
    Dataset(
        name="oil_gas_field_rules",
        category="manual",
        url="https://www.rrc.texas.gov/media/01wfdxlx/oilgasfieldrules.pdf",
        filename="oil_gas_field_rules.pdf",
        description="Field rules: spacing and production rules per field",
    ),
    Dataset(
        name="statewide_field_data",
        category="manual",
        url="https://www.rrc.texas.gov/media/sn4f1nyy/field_information_oga090.pdf",
        filename="oga090_statewide_field_data.pdf",
        description="Statewide field data: rules, allowables, production",
    ),
    Dataset(
        name="gas_masters",
        category="manual",
        url="https://www.rrc.texas.gov/media/jukdy4oh/gsa020k.pdf",
        filename="gsa020k_gas_masters.pdf",
        description="Gas Masters: 12 months production + 14 months allowable per gas well",
    ),
    Dataset(
        name="oil_masters",
        category="manual",
        url="https://www.rrc.texas.gov/media/5mlpe0t2/ola013k.pdf",
        filename="ola013k_oil_masters.pdf",
        description="Oil Masters: 12 months production + 14 months allowable per oil lease",
    ),
    Dataset(
        name="pr_gas_disposition",
        category="manual",
        url="https://www.rrc.texas.gov/media/rovpac0l/oga0861.pdf",
        filename="oga0861_pr_gas_disposition.pdf",
        description="PR (P1/P2) Gas Disposition layout",
    ),
    Dataset(
        name="historical_ledger",
        category="manual",
        url="https://www.rrc.texas.gov/media/ocwhbdp2/oilandgashistoricalledger_lad001.pdf",
        filename="lad001_historical_ledger.pdf",
        description="Historical Ledger: lease-level production since 1993",
    ),
    Dataset(
        name="pdq_dump",
        category="manual",
        url="https://www.rrc.texas.gov/media/50ypu2cg/pdq-dump-user-manual.pdf",
        filename="pdq_dump_manual.pdf",
        description="Production Data Query Dump user manual",
    ),
    Dataset(
        name="pr_pending_lease",
        category="manual",
        url="https://www.rrc.texas.gov/media/awhgplsn/prpendinglease.pdf",
        filename="pr_pending_lease.pdf",
        description="Production Report for pending leases",
    ),
    Dataset(
        name="statewide_production",
        category="manual",
        url="https://www.rrc.texas.gov/media/0a4dqvag/pda001.pdf",
        filename="pda001_statewide_production.pdf",
        description="Statewide production data layout",
    ),
    Dataset(
        name="p18_skim_oil",
        category="manual",
        url="https://www.rrc.texas.gov/media/j20f30cs/p-18_user_guide.pdf",
        filename="p18_skim_oil_condensate.pdf",
        description="P-18 Skim Oil/Condensate Report user guide",
    ),
    Dataset(
        name="p4_certificate",
        category="manual",
        url="https://www.rrc.texas.gov/media/wdmjsoph/p4-user-manual_p4a002_feb2015.pdf",
        filename="p4_certificate_authorization.pdf",
        description="P-4 Certificate of Authorization (Producer's Transportation Authority)",
    ),
    Dataset(
        name="oil_gas_docket",
        category="manual",
        url="https://www.rrc.texas.gov/media/spfpwolf/oda037k.pdf",
        filename="oda037k_oil_gas_docket.pdf",
        description="Oil and Gas Docket data layout",
    ),
    Dataset(
        name="p5_organization",
        category="manual",
        url="https://www.rrc.texas.gov/media/jtqfynn3/ora001_p5_manual_october-2014.pdf",
        filename="ora001_p5_organization.pdf",
        description="P-5 Organization Report data layout",
    ),
    Dataset(
        name="r3_gas_processing",
        category="manual",
        url="https://www.rrc.texas.gov/media/glnde44t/gra001.pdf",
        filename="gra001_r3_gas_processing_plants.pdf",
        description="R-3 Gas Processing Plants Monthly Report",
    ),
    Dataset(
        name="r3_dataload",
        category="manual",
        url="https://www.rrc.texas.gov/media/egmakucr/r-3-dataload-definition.pdf",
        filename="r3_dataload_definition.pdf",
        description="R-3 dataload definition (current JSON format spec)",
    ),
    Dataset(
        name="inspections_violations",
        category="manual",
        url="https://www.rrc.texas.gov/media/kv3nyrub/inspections-and-violations-data-v10.pdf",
        filename="inspections_violations_v10.pdf",
        description="RRC inspections and violations data layout",
    ),
    Dataset(
        name="wellbore_database",
        category="manual",
        url="https://www.rrc.texas.gov/media/cx2i4xrm/wba091_well-bore-database.pdf",
        filename="wba091_wellbore_database.pdf",
        description="Full Wellbore database layout",
    ),
    Dataset(
        name="wellbore_query",
        category="manual",
        url="https://www.rrc.texas.gov/media/di1mm5or/og_wellbore_ewadefinitionmanual2013-10-30_subscription.pdf",
        filename="wellbore_query.pdf",
        description="Wellbore query data layout (EWA definition)",
    ),
    Dataset(
        name="statewide_well",
        category="manual",
        url="https://www.rrc.texas.gov/media/oyajan55/wla001k.pdf",
        filename="wla001k_statewide_well.pdf",
        description="Statewide Oil/Gas Well Database",
    ),
    Dataset(
        name="oil_detail_well",
        category="manual",
        url="https://www.rrc.texas.gov/media/f5ll4hom/ola028k.pdf",
        filename="ola028k_oil_detail_well.pdf",
        description="Oil Detail Well data layout, including SWR 14(B)(2)",
    ),
    Dataset(
        name="oil_well_status_w10",
        category="manual",
        url="https://www.rrc.texas.gov/media/htcn0kpc/ola001k.pdf",
        filename="ola001k_oil_well_status_w10.pdf",
        description="Oil Well Status Report (Form W-10) layout",
    ),
    Dataset(
        name="gas_well_status_g10",
        category="manual",
        url="https://www.rrc.texas.gov/media/frhd4zs4/gsa018k.pdf",
        filename="gsa018k_gas_well_status_g10.pdf",
        description="Gas Well Status Report (Form G-10) layout",
    ),
    Dataset(
        name="completion_subscriptions",
        category="manual",
        url="https://www.rrc.texas.gov/media/ltudbyql/cmpldatasubscriptions_usermanual_20171101.pdf",
        filename="completion_data_subscriptions.pdf",
        description="Completion Information data layout (G-1, W-2, etc.)",
    ),
    Dataset(
        name="high_cost_gas",
        category="manual",
        url="https://www.rrc.texas.gov/media/nmke5wqe/highcostgasleases.pdf",
        filename="high_cost_gas.pdf",
        description="High Cost Gas severance tax incentive data",
    ),
    Dataset(
        name="high_cost_gas_tight",
        category="manual",
        url="https://www.rrc.texas.gov/media/503hufzj/gasleaseswithinhighcostgas.pdf",
        filename="high_cost_gas_tight_sands.pdf",
        description="High Cost Gas - Tight Sands only",
    ),
    Dataset(
        name="ngpa",
        category="manual",
        url="https://www.rrc.texas.gov/media/td0dz2ta/ngpausermanual.pdf",
        filename="ngpa_user_manual.pdf",
        description="Natural Gas Policy Act data layout",
    ),
    Dataset(
        name="st1_user_guide",
        category="manual",
        url="https://www.rrc.texas.gov/media/xwylw1uh/st-1-online-system-reporting-user-guide-formatted.pdf",
        filename="st1_application_report.pdf",
        description="ST-1 Application Report user guide",
    ),
    Dataset(
        name="uic_manual",
        category="manual",
        url="https://www.rrc.texas.gov/media/v3onmigl/uic_manual_uia010_3116.pdf",
        filename="uia010_uic_database.pdf",
        description="Underground Injection Control (UIC) Database",
    ),
    Dataset(
        name="digital_map_user_guide",
        category="manual",
        url="https://www.rrc.texas.gov/media/kmld3uzj/digital-map-information-user-guide.pdf",
        filename="digital_map_user_guide.pdf",
        description="Digital map information user guide",
    ),
    Dataset(
        name="well_api_manual",
        category="manual",
        url="https://www.rrc.texas.gov/media/bkyl5qvx/well-api-manual.pdf",
        filename="well_api_manual.pdf",
        description="Statewide API data layout (well identifiers)",
    ),
]


# Structured data files (ASCII / CSV / EBCDIC).
# Note: many of these are EBCDIC and need conversion. Manuals are in MANUALS above.
STRUCTURED_DATA: list[Dataset] = [
    Dataset(
        name="inspections_violations_data",
        category="structured",
        url="https://mft.rrc.texas.gov/link/c7c28dc9-b218-4f0a-8278-bf15d009def1",
        filename="inspections_violations.txt",
        description="RRC inspections and violations data (TXT, weekly updates)",
    ),
    Dataset(
        name="oil_gas_field_rules_ascii",
        category="structured",
        url="https://mft.rrc.texas.gov/link/6b12ba15-d46a-46a9-b0ce-ca743d9cefe6",
        filename="oil_gas_field_rules.txt",
        description="Oil & Gas Field Rules in ASCII format",
    ),
    Dataset(
        name="oil_gas_field_name_numbers_ascii",
        category="structured",
        url="https://mft.rrc.texas.gov/link/3122a5ec-eb3b-4ed2-908b-f41fa94ab8ba",
        filename="oil_gas_field_name_numbers.txt",
        description="Oil & Gas field name and numbers (ASCII)",
    ),
    Dataset(
        name="oil_gas_docket_ascii",
        category="structured",
        url="https://mft.rrc.texas.gov/link/e9af053b-28c8-49b2-ad40-bf2f10f4f21a",
        filename="oil_gas_docket.txt",
        description="Oil and Gas Docket in ASCII format",
    ),
    Dataset(
        name="horizontal_drilling_permits_ascii",
        category="structured",
        url="https://mft.rrc.texas.gov/link/c725637f-6748-47b9-ad74-e0396879d88b",
        filename="horizontal_drilling_permits.txt",
        description="Horizontal drilling permits (ASCII)",
    ),
    Dataset(
        name="pdq_dump_csv",
        category="structured",
        url="https://mft.rrc.texas.gov/link/1f5ddb8d-329a-4459-b7f8-177b4f5ee60d",
        filename="production_data_query_dump.csv",
        description="Production Data Query Dump (CSV, 1993 to current)",
    ),
    Dataset(
        name="st1_application_csv",
        category="structured",
        url="https://mft.rrc.texas.gov/link/cd6842db-a690-432b-94b7-df10a0b44166",
        filename="st1_application_report.csv",
        description="ST-1 Application Report (CSV)",
    ),
]


# Imaged form PDFs - large, optional.
IMAGED_DATA: list[Dataset] = [
    Dataset(
        name="w1_imaged_permits",
        category="imaged",
        url="https://mft.rrc.texas.gov/link/f11363bb-8120-4e8c-bbc0-a253ec0a85d4",
        filename="w1_imaged_permits.zip",
        description="Form W-1 (Application for Permit to Drill) imaged PDFs, organized by district",
    ),
    Dataset(
        name="completion_imaged",
        category="imaged",
        url="https://mft.rrc.texas.gov/link/8e91acb8-69cc-4d57-ad72-c7f7d5a7675e",
        filename="completion_imaged_files.zip",
        description="Completion form images (G-1, W-2, P-4, etc.)",
    ),
    Dataset(
        name="directional_surveys",
        category="imaged",
        url="https://mft.rrc.texas.gov/link/01769aa7-dee8-4121-bb25-e7557307f6bd",
        filename="directional_survey_applications.zip",
        description="Directional Survey Applications (daily PDFs)",
    ),
]


def download_one(dataset: Dataset, target_dir: Path, force: bool = False) -> tuple[Dataset, bool, str]:
    """Download a single dataset. Returns (dataset, success, message)."""
    target = target_dir / dataset.category / dataset.filename
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        return dataset, True, f"exists ({target.stat().st_size:,} bytes)"

    try:
        headers = {"User-Agent": USER_AGENT}
        with requests.get(dataset.url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            sha = hashlib.sha256()
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "wb") as out:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    out.write(chunk)
                    sha.update(chunk)

            tmp.rename(target)
            size = target.stat().st_size
            checksum = sha.hexdigest()
            return dataset, True, f"downloaded {size:,} bytes (sha256 {checksum[:12]}...)"

    except Exception as exc:
        return dataset, False, f"FAILED: {type(exc).__name__}: {exc}"


def download_set(datasets: list[Dataset], target_dir: Path, workers: int = 4, force: bool = False) -> None:
    print(f"\nDownloading {len(datasets)} files to {target_dir}/")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, ds, target_dir, force): ds for ds in datasets}

        ok = 0
        failed = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Files"):
            dataset, success, message = future.result()
            if success:
                ok += 1
                tqdm.write(f"  OK  {dataset.name}: {message}")
            else:
                failed.append((dataset, message))
                tqdm.write(f"  ERR {dataset.name}: {message}")

    print(f"\nDone. {ok}/{len(datasets)} successful.")
    if failed:
        print("\nFailures:")
        for ds, msg in failed:
            print(f"  - {ds.name}: {msg}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-data", action="store_true", help="Also download structured data files")
    parser.add_argument("--include-all", action="store_true", help="Download everything including large imaged files")
    parser.add_argument("--target-dir", type=Path, default=RAW_DIR, help="Where to put the files")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)

    print(f"TROA data download")
    print(f"  Target: {args.target_dir.absolute()}")
    print(f"  Manuals: {len(MANUALS)}")

    download_set(MANUALS, args.target_dir, args.workers, args.force)

    if args.include_data or args.include_all:
        print(f"\n  Structured data: {len(STRUCTURED_DATA)}")
        download_set(STRUCTURED_DATA, args.target_dir, args.workers, args.force)

    if args.include_all:
        print(f"\n  Imaged data: {len(IMAGED_DATA)}")
        print("  WARNING: these are multi-gigabyte zip files; expect long downloads.")
        download_set(IMAGED_DATA, args.target_dir, args.workers, args.force)

    print(f"\nNext step:")
    print(f"  python -m src.ingest.pipeline --corpus {args.target_dir}/manual/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial downloads are kept; rerun to resume (existing files skipped).")
        sys.exit(130)
