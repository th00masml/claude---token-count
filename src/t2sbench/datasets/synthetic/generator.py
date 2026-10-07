"""Deterministic generator of the synthetic legacy production database (SQLite).

The schema imitates an old shop-floor system: short Polish abbreviations as table and
column names and single-letter status codes. Code values come from ``codes.yaml``.
All randomness flows from one ``random.Random(seed)`` so the same seed gives a
byte-identical set of rows.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

HERE = Path(__file__).parent
CODES_PATH = HERE / "codes.yaml"
SCHEMA_DOC_PATH = HERE / "schema_doc.yaml"

START_DATE = date(2026, 1, 1)
N_DAYS = 90  # 2026-01-01 .. 2026-03-31
REPORT_DATE = date(2026, 3, 31)  # "today" from the point of view of order statuses

DDL = """
CREATE TABLE LIN (
    LIN_ID    INTEGER PRIMARY KEY,
    LIN_KOD   TEXT NOT NULL,
    LIN_NAZ   TEXT NOT NULL,
    WYDZ      TEXT NOT NULL,
    LIN_STAT  TEXT NOT NULL
);
CREATE TABLE MCH (
    MCH_ID    INTEGER PRIMARY KEY,
    MCH_KOD   TEXT NOT NULL,
    MCH_NAZ   TEXT NOT NULL,
    LIN_ID    INTEGER NOT NULL REFERENCES LIN(LIN_ID),
    MCH_TYP   TEXT NOT NULL,
    MCH_AKT   TEXT NOT NULL,
    ROK_PROD  INTEGER
);
CREATE TABLE MCH_STAT (
    MS_ID     INTEGER PRIMARY KEY,
    MCH_ID    INTEGER NOT NULL REFERENCES MCH(MCH_ID),
    TS        TEXT NOT NULL,
    STAT      TEXT NOT NULL
);
CREATE TABLE ZMIANA (
    ZM_ID     INTEGER PRIMARY KEY,
    ZM_DATA   TEXT NOT NULL,
    ZM_KOD    TEXT NOT NULL,
    BRYG      TEXT NOT NULL,
    TS_OD     TEXT NOT NULL,
    TS_DO     TEXT NOT NULL
);
CREATE TABLE PRZYCZ (
    PRZYCZ_KOD TEXT PRIMARY KEY,
    OPIS       TEXT NOT NULL,
    KAT        TEXT NOT NULL
);
CREATE TABLE PRZEST (
    PRZ_ID     INTEGER PRIMARY KEY,
    MCH_ID     INTEGER NOT NULL REFERENCES MCH(MCH_ID),
    ZM_ID      INTEGER NOT NULL REFERENCES ZMIANA(ZM_ID),
    TS_OD      TEXT NOT NULL,
    TS_DO      TEXT NOT NULL,
    CZAS_MIN   INTEGER NOT NULL,
    PRZYCZ_KOD TEXT NOT NULL REFERENCES PRZYCZ(PRZYCZ_KOD),
    PLAN_FL    TEXT NOT NULL,
    UWAGI      TEXT
);
CREATE TABLE PROD (
    PROD_ID    INTEGER PRIMARY KEY,
    INDEKS     TEXT NOT NULL,
    NAZWA      TEXT NOT NULL,
    JM         TEXT NOT NULL,
    GRUPA      TEXT NOT NULL,
    PROD_STAT  TEXT NOT NULL,
    MASA_KG    REAL
);
CREATE TABLE ZLEC (
    ZLEC_ID    INTEGER PRIMARY KEY,
    ZLEC_NR    TEXT NOT NULL,
    PROD_ID    INTEGER NOT NULL REFERENCES PROD(PROD_ID),
    ZLEC_ILOSC INTEGER NOT NULL,
    ZLEC_STAT  TEXT NOT NULL,
    PRIOR      TEXT NOT NULL,
    DT_UTW     TEXT NOT NULL,
    DT_PLAN_OD TEXT NOT NULL,
    DT_PLAN_DO TEXT NOT NULL
);
CREATE TABLE ZLEC_POZ (
    POZ_ID     INTEGER PRIMARY KEY,
    ZLEC_ID    INTEGER NOT NULL REFERENCES ZLEC(ZLEC_ID),
    POZ_NR     INTEGER NOT NULL,
    OPER_KOD   TEXT NOT NULL,
    LIN_ID     INTEGER NOT NULL REFERENCES LIN(LIN_ID),
    POZ_ILOSC  INTEGER NOT NULL,
    POZ_STAT   TEXT NOT NULL
);
CREATE TABLE OPER (
    OPER_ID    INTEGER PRIMARY KEY,
    NR_EWID    TEXT NOT NULL,
    IMIE       TEXT NOT NULL,
    NAZW       TEXT NOT NULL,
    KWAL       TEXT NOT NULL,
    OPER_STAT  TEXT NOT NULL,
    LIN_ID     INTEGER REFERENCES LIN(LIN_ID),
    DT_ZATR    TEXT NOT NULL
);
CREATE TABLE OPER_ZM (
    OZ_ID      INTEGER PRIMARY KEY,
    OPER_ID    INTEGER NOT NULL REFERENCES OPER(OPER_ID),
    ZM_ID      INTEGER NOT NULL REFERENCES ZMIANA(ZM_ID),
    MCH_ID     INTEGER NOT NULL REFERENCES MCH(MCH_ID)
);
CREATE TABLE PROD_REJ (
    REJ_ID     INTEGER PRIMARY KEY,
    ZM_ID      INTEGER NOT NULL REFERENCES ZMIANA(ZM_ID),
    MCH_ID     INTEGER NOT NULL REFERENCES MCH(MCH_ID),
    ZLEC_ID    INTEGER NOT NULL REFERENCES ZLEC(ZLEC_ID),
    POZ_ID     INTEGER NOT NULL REFERENCES ZLEC_POZ(POZ_ID),
    OPER_ID    INTEGER NOT NULL REFERENCES OPER(OPER_ID),
    ILOSC_OK   INTEGER NOT NULL,
    ILOSC_BR   INTEGER NOT NULL,
    TS_REJ     TEXT NOT NULL
);
CREATE TABLE BR_TYP (
    BR_TYP_KOD TEXT PRIMARY KEY,
    OPIS       TEXT NOT NULL,
    KAT        TEXT NOT NULL
);
CREATE TABLE BRAKI (
    BR_ID      INTEGER PRIMARY KEY,
    REJ_ID     INTEGER NOT NULL REFERENCES PROD_REJ(REJ_ID),
    BR_TYP_KOD TEXT NOT NULL REFERENCES BR_TYP(BR_TYP_KOD),
    ILOSC      INTEGER NOT NULL,
    BR_STAT    TEXT NOT NULL,
    TS_ZGL     TEXT NOT NULL
);
"""

# line code, name, department, status, machine types
LINES = [
    ("L01", "LINIA CIECIA I TLOCZENIA 1", "M", "A", "CCPP"),
    ("L02", "LINIA CIECIA I TLOCZENIA 2", "M", "A", "CPP"),
    ("L03", "LINIA SPAWALNICZA 1", "S", "A", "WWWK"),
    ("L04", "LINIA SPAWALNICZA 2", "S", "A", "WWK"),
    ("L05", "LAKIERNIA PROSZKOWA", "P", "A", "LLK"),
    ("L06", "LINIA MONTAZU KONCOWEGO", "A", "A", "MMMK"),
    ("L07", "LINIA SPAWALNICZA 3 (STARA)", "S", "N", "W"),
]
MCH_NAMES = {
    "C": "LASER CO2", "P": "PRASA MIMOSRODOWA", "W": "ROBOT SPAW. MIG",
    "L": "KABINA LAK.", "M": "STANOWISKO MONT.", "K": "STANOWISKO KJ",
}
# machine type -> operation code performed on it
TYP_TO_OPER = {"C": "CI", "P": "TL", "W": "SP", "L": "LK", "M": "MT", "K": "KJ"}
# operation -> departments whose lines can do it
OPER_TO_WYDZ = {"CI": "M", "TL": "M", "SP": "S", "LK": "P", "MT": "A", "KJ": "SA"}
# product group -> routing (sequence of operations)
ROUTING = {
    "A": ["CI", "TL", "LK", "KJ"],
    "B": ["CI", "SP", "LK", "KJ"],
    "C": ["CI", "TL", "SP", "LK", "KJ"],
    "D": ["SP", "MT", "KJ"],
}
PRODUCT_NAMES = {
    "A": ["WSPORNIK", "MOCOWANIE", "KATOWNIK", "UCHWYT", "OBEJMA"],
    "B": ["RAMA", "RAMA POMOCNICZA", "STELAZ", "BELKA", "PODSTAWA"],
    "C": ["OBUDOWA", "OSLONA", "POKRYWA", "PANEL", "DRZWICZKI"],
    "D": ["ZESPOL ZAWIASU", "ZESPOL NAPEDU", "MODUL WSPORCZY", "ZESPOL RAMY", "KOMPLET MONTAZOWY"],
}
SHIFT_HOURS = {"1": (6, 14), "2": (14, 22), "3": (22, 30)}

FIRST_NAMES = ["Adam", "Barbara", "Cezary", "Dorota", "Edward", "Fabian", "Grazyna", "Henryk",
               "Irena", "Jacek", "Kamil", "Lucyna", "Marek", "Natalia", "Olaf", "Paulina",
               "Robert", "Sylwia", "Tomasz", "Urszula", "Wiktor", "Zofia"]
LAST_NAMES = ["Kowalczyk", "Nowicki", "Wrobel", "Mazur", "Krawiec", "Pawlak", "Dudek", "Zajac",
              "Kaczmarek", "Sikora", "Baran", "Szewczyk", "Ostrowski", "Witkowski", "Malinowski",
              "Jablonski", "Kubiak", "Wojcik", "Lis", "Gorski", "Michalak", "Sadowski"]

# reason code -> (relative weight, min minutes, max minutes)
DOWNTIME_PROFILE = {
    "AWM": (10, 15, 240), "AWE": (7, 10, 180), "AWH": (4, 20, 300), "AWS": (5, 10, 120),
    "BMT": (9, 10, 150), "BOP": (4, 30, 240), "OCZ": (6, 10, 90), "JKP": (4, 20, 180),
    "PRZ": (12, 15, 90), "KON": (5, 60, 240), "PRZW": (14, 30, 30),
}
# reason code -> machine status logged while the machine is down
REASON_TO_MCH_STAT = {"AWM": "A", "AWE": "A", "AWH": "A", "AWS": "A", "PRZ": "X", "KON": "K"}
# scrap type -> machine types where it can occur
SCRAP_BY_TYP = {
    "C": ["MPK", "WWY", "WOT"], "P": ["MPK", "PGI", "WWY", "WOT"],
    "W": ["PSP", "PGI", "MWT", "WWY"], "L": ["LZC", "LRY", "LKO"],
    "M": ["PNZ", "WOT", "LRY"], "K": ["WWY", "LRY", "PSP", "PNZ"],
}
BASE_OUTPUT = {"C": 220, "P": 260, "W": 90, "L": 140, "M": 60, "K": 180}


def load_codes(path: Path = CODES_PATH) -> dict[str, dict[str, dict[str, str]]]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {col: {str(k): v for k, v in vals.items()} for col, vals in raw.items()}


def _ts(d: date, hour: int, minute: int = 0) -> str:
    dt = datetime(d.year, d.month, d.day) + timedelta(hours=hour, minutes=minute)
    return dt.strftime("%Y-%m-%d %H:%M")


def _add_min(ts: str, minutes: int) -> str:
    return (datetime.strptime(ts, "%Y-%m-%d %H:%M") + timedelta(minutes=minutes)).strftime(
        "%Y-%m-%d %H:%M"
    )


def generate(db_path: str | Path, seed: int = 42) -> dict[str, int]:
    """Create the database at ``db_path`` (overwrites). Returns row counts per table."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    rng = random.Random(seed)
    codes = load_codes()

    rows: dict[str, list[tuple]] = {}

    # --- dictionaries -------------------------------------------------------
    rows["PRZYCZ"] = [
        (k, _legacy(v["pl"]), _reason_category(k)) for k, v in codes["PRZYCZ.PRZYCZ_KOD"].items()
    ]
    rows["BR_TYP"] = [
        (k, _legacy(v["pl"]), k[0]) for k, v in codes["BR_TYP.BR_TYP_KOD"].items()
    ]

    # --- lines and machines -------------------------------------------------
    rows["LIN"], rows["MCH"] = [], []
    mch_id = 0
    for lin_id, (kod, naz, wydz, stat, types) in enumerate(LINES, start=1):
        rows["LIN"].append((lin_id, kod, naz, wydz, stat))
        for i, t in enumerate(types, start=1):
            mch_id += 1
            akt = "T" if stat == "A" else "N"
            rows["MCH"].append(
                (mch_id, f"{t}{kod[1:]}{i:02d}", f"{MCH_NAMES[t]} {i}", lin_id, t, akt,
                 rng.randint(1998, 2022))
            )
    lin_by_id = {r[0]: r for r in rows["LIN"]}
    active_mch = [m for m in rows["MCH"] if m[5] == "T"]

    # --- shifts -------------------------------------------------------------
    rows["ZMIANA"] = []
    shifts = []  # (zm_id, date, kod)
    zm_id = 0
    for d_i in range(N_DAYS):
        d = START_DATE + timedelta(days=d_i)
        for s_i, kod in enumerate(["1", "2", "3"]):
            zm_id += 1
            bryg = "ABCD"[(d_i // 2 + s_i) % 4]
            h0, h1 = SHIFT_HOURS[kod]
            rows["ZMIANA"].append((zm_id, d.isoformat(), kod, bryg, _ts(d, h0), _ts(d, h1)))
            shifts.append((zm_id, d, kod))

    # --- products -----------------------------------------------------------
    rows["PROD"] = []
    prod_id = 0
    for grupa in "ABCD":
        for base in PRODUCT_NAMES[grupa]:
            for variant in range(2):
                prod_id += 1
                jm = "KPL" if grupa == "D" else ("KG" if rng.random() < 0.1 else "SZT")
                stat = "W" if rng.random() < 0.12 else "A"
                rows["PROD"].append(
                    (prod_id, f"{grupa}{rng.randint(10, 99)}-{prod_id:04d}",
                     f"{base} {rng.choice(['S', 'M', 'L', 'XL'])}{variant + 1}", jm, grupa, stat,
                     round(rng.uniform(0.2, 35.0), 2))
                )
    active_prod = [p for p in rows["PROD"] if p[5] == "A"]

    # --- operators ----------------------------------------------------------
    rows["OPER"] = []
    used_names = set()
    oper_by_line: dict[int, list[int]] = {}
    for oper_id in range(1, 67):
        while True:
            name = (rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES))
            if name not in used_names:
                used_names.add(name)
                break
        lin_id = (oper_id - 1) % 6 + 1  # active lines only
        r = rng.random()
        stat = "Z" if r < 0.06 else ("U" if r < 0.12 else "A")
        kwal = rng.choices(["1", "2", "3"], weights=[3, 6, 2])[0]
        dt_zatr = (date(2004, 1, 1) + timedelta(days=rng.randint(0, 7900))).isoformat()
        rows["OPER"].append((oper_id, f"E{4000 + oper_id * 7}", name[0], name[1], kwal, stat,
                             lin_id, dt_zatr))
        if stat != "Z":
            oper_by_line.setdefault(lin_id, []).append(oper_id)

    # --- orders and order positions -----------------------------------------
    rows["ZLEC"], rows["ZLEC_POZ"] = [], []
    lines_by_wydz: dict[str, list[int]] = {}
    for r in rows["LIN"]:
        if r[4] == "A":
            lines_by_wydz.setdefault(r[3], []).append(r[0])
    poz_id = 0
    for zlec_id in range(1, 421):
        prod = rng.choice(active_prod)
        start = START_DATE + timedelta(days=rng.randint(-10, 115))
        dur = rng.randint(3, 21)
        end = start + timedelta(days=dur)
        created = start - timedelta(days=rng.randint(3, 20))
        if end < REPORT_DATE - timedelta(days=5):
            stat = rng.choices(["Z", "A"], weights=[93, 7])[0]
        elif start <= REPORT_DATE:
            stat = rng.choices(["R", "W", "A"], weights=[78, 15, 7])[0]
        else:
            stat = rng.choices(["N", "A"], weights=[94, 6])[0]
        prior = rng.choices(["1", "2", "3"], weights=[2, 6, 2])[0]
        qty = rng.choice([50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500])
        rows["ZLEC"].append(
            (zlec_id, f"ZP/{start.year}/{zlec_id:05d}", prod[0], qty, stat, prior,
             created.isoformat(), start.isoformat(), end.isoformat())
        )
        for nr, oper in enumerate(ROUTING[prod[4]], start=1):
            poz_id += 1
            wydz_options = [lin for w in OPER_TO_WYDZ[oper] for lin in lines_by_wydz.get(w, [])]
            lin_id = rng.choice(wydz_options)
            if stat == "Z":
                pstat = "Z"
            elif stat in ("N", "A"):
                pstat = "O"
            else:
                pstat = rng.choice(["O", "R", "Z"])
            rows["ZLEC_POZ"].append((poz_id, zlec_id, nr, oper, lin_id, qty, pstat))

    zlec_by_id = {z[0]: z for z in rows["ZLEC"]}
    # (lin_id, oper) -> [(poz_id, zlec_id, start, end)]
    poz_index: dict[tuple[int, str], list[tuple[int, int, date, date]]] = {}
    for p in rows["ZLEC_POZ"]:
        z = zlec_by_id[p[1]]
        if z[4] in ("R", "W", "Z"):
            poz_index.setdefault((p[4], p[3]), []).append(
                (p[0], p[1], date.fromisoformat(z[7]), date.fromisoformat(z[8]))
            )

    # --- production, scrap, operator assignment ----------------------------
    rows["PROD_REJ"], rows["BRAKI"], rows["OPER_ZM"] = [], [], []
    rej_id = br_id = oz_id = 0
    for zm, d, kod in shifts:
        h0, _ = SHIFT_HOURS[kod]
        for m in active_mch:
            if rng.random() > 0.62:
                continue
            oper_kod = TYP_TO_OPER[m[4]]
            candidates = [c for c in poz_index.get((m[3], oper_kod), []) if c[2] <= d <= c[3]]
            if not candidates:
                continue
            poz, zlec, _, _ = rng.choice(candidates)
            operators = oper_by_line.get(m[3], [])
            if not operators:
                continue
            oper_id = rng.choice(operators)
            base = BASE_OUTPUT[m[4]] * (0.8 if kod == "3" else 1.0)
            ok = max(5, int(rng.gauss(base, base * 0.18)))
            rej_id += 1
            ts_rej = _ts(d, h0 + 7, rng.randint(30, 59))
            scrap_total = 0
            if rng.random() < 0.38:
                for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
                    br_id += 1
                    qty = max(1, int(rng.expovariate(1 / (base * 0.025))))
                    bstat = rng.choices(["P", "Z", "O"], weights=[70, 18, 12])[0]
                    if bstat != "O":
                        scrap_total += qty
                    rows["BRAKI"].append(
                        (br_id, rej_id, rng.choice(SCRAP_BY_TYP[m[4]]), qty, bstat,
                         _ts(d, h0 + rng.randint(1, 7), rng.randint(0, 59)))
                    )
            rows["PROD_REJ"].append((rej_id, zm, m[0], zlec, poz, oper_id, ok, scrap_total, ts_rej))
            oz_id += 1
            rows["OPER_ZM"].append((oz_id, oper_id, zm, m[0]))

    # --- downtimes and machine status log ----------------------------------
    rows["PRZEST"], rows["MCH_STAT"] = [], []
    reasons = list(DOWNTIME_PROFILE)
    weights = [DOWNTIME_PROFILE[r][0] for r in reasons]
    prz_id = ms_id = 0
    for zm, d, kod in shifts:
        h0, _ = SHIFT_HOURS[kod]
        for m in active_mch:
            if rng.random() > 0.27:
                continue
            reason = rng.choices(reasons, weights=weights)[0]
            _, lo, hi = DOWNTIME_PROFILE[reason]
            minutes = rng.randint(lo, hi)
            start_min = rng.randint(0, max(0, 8 * 60 - minutes))
            ts_od = _ts(d, h0, 0)
            ts_od = _add_min(ts_od, start_min)
            ts_do = _add_min(ts_od, minutes)
            plan = "T" if _reason_category(reason) == "P" else "N"
            note = rng.choice([None, None, None, "ZGLOSZONO DO UR", "CZEKA NA CZESC", "PATRZ RAPORT"])
            prz_id += 1
            rows["PRZEST"].append((prz_id, m[0], zm, ts_od, ts_do, minutes, reason, plan, note))
            ms_id += 1
            rows["MCH_STAT"].append((ms_id, m[0], ts_od, REASON_TO_MCH_STAT.get(reason, "S")))
            ms_id += 1
            rows["MCH_STAT"].append((ms_id, m[0], ts_do, "P"))

    _write(db_path, rows)
    return {t: len(r) for t, r in rows.items()}


def _legacy(text: str) -> str:
    """Upper case without Polish diacritics, as an old system would store it."""
    table = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    return text.translate(table).upper()


def _reason_category(code: str) -> str:
    if code.startswith("AW"):
        return "T"
    if code in ("BMT", "BOP", "OCZ"):
        return "O"
    if code == "JKP":
        return "J"
    return "P"  # PRZ, KON, PRZW


def _write(db_path: Path, rows: dict[str, list[tuple]]) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(DDL)
        for table, data in rows.items():
            if not data:
                continue
            ph = ",".join("?" * len(data[0]))
            con.executemany(f"INSERT INTO {table} VALUES ({ph})", data)
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(generate(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/prod.sqlite"))
