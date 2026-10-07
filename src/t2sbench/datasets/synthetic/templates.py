"""Intent templates for questions about the synthetic production DB.

Each template has a slot sampler, PL and EN phrasings (index-aligned, so variant i in
PL is the translation of variant i in EN) and a gold SQL pattern. Templates marked
``split="test"`` are used only for the 40-question test set; ``split="train"`` only for
the LoRA training / validation pool. Test and train templates have different intents.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property

from t2sbench.datasets.synthetic.generator import load_codes

MONTH_EN = {1: "January", 2: "February", 3: "March", 4: "April"}
MONTH_PL_GEN = {1: "stycznia", 2: "lutego", 3: "marca", 4: "kwietnia"}
MONTH_PL_LOC = {1: "styczniu", 2: "lutym", 3: "marcu", 4: "kwietniu"}
SHIFT_PL = {"1": "porannej", "2": "popołudniowej", "3": "nocnej"}
SHIFT_EN = {"1": "morning", "2": "afternoon", "3": "night"}


@dataclass(frozen=True)
class Template:
    id: str
    split: str  # "test" | "train"
    intent: str
    sample: Callable[["Ctx", random.Random], dict]
    pl: tuple[str, ...]
    en: tuple[str, ...]
    sql: str
    # for ORDER BY ... LIMIT k templates: (index of sort-key column, k). The builder checks
    # that the top k are strictly ordered and the k-th differs from the (k+1)-th, so the
    # gold answer does not depend on how ties are broken.
    order_check: tuple[int, int] | None = None
    allow_zero: bool = False

    def __post_init__(self):
        assert len(self.pl) == len(self.en), self.id


class Ctx:
    """Lookup lists drawn from the generated DB, used by slot samplers."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.codes = load_codes()

    def q(self, sql: str, *args) -> list[tuple]:
        return self.con.execute(sql, args).fetchall()

    @cached_property
    def lines(self) -> list[str]:
        return [r[0] for r in self.q("SELECT LIN_KOD FROM LIN WHERE LIN_STAT='A' ORDER BY LIN_KOD")]

    @cached_property
    def machines(self) -> list[str]:
        return [r[0] for r in self.q("SELECT MCH_KOD FROM MCH WHERE MCH_AKT='T' ORDER BY MCH_KOD")]

    @cached_property
    def dates(self) -> list[str]:
        return [r[0] for r in self.q("SELECT DISTINCT ZM_DATA FROM ZMIANA ORDER BY ZM_DATA")]

    @cached_property
    def orders_with_output(self) -> list[str]:
        return [r[0] for r in self.q(
            "SELECT DISTINCT z.ZLEC_NR FROM ZLEC z JOIN PROD_REJ r ON r.ZLEC_ID=z.ZLEC_ID ORDER BY 1")]

    @cached_property
    def orders(self) -> list[str]:
        return [r[0] for r in self.q("SELECT ZLEC_NR FROM ZLEC ORDER BY ZLEC_ID")]

    @cached_property
    def products(self) -> list[str]:
        return [r[0] for r in self.q(
            "SELECT DISTINCT p.INDEKS FROM PROD p JOIN ZLEC z ON z.PROD_ID=p.PROD_ID ORDER BY 1")]

    @cached_property
    def prod_reports(self) -> list[tuple]:
        return self.q(
            "SELECT l.LIN_KOD, m.MCH_KOD, z.ZM_DATA, z.ZM_KOD FROM PROD_REJ r "
            "JOIN MCH m ON m.MCH_ID=r.MCH_ID JOIN LIN l ON l.LIN_ID=m.LIN_ID "
            "JOIN ZMIANA z ON z.ZM_ID=r.ZM_ID ORDER BY r.REJ_ID")

    def meaning(self, column: str, code: str, lang: str) -> str:
        return self.codes[column][code][lang]


# ---------------------------------------------------------------- slot helpers

def _date_slots(d: str, prefix: str = "date") -> dict:
    x = date.fromisoformat(d)
    return {
        prefix: d,
        f"{prefix}_pl": f"{x.day} {MONTH_PL_GEN[x.month]} {x.year}",
        f"{prefix}_en": f"{MONTH_EN[x.month]} {x.day}, {x.year}",
    }


def _month_slots(m: int) -> dict:
    nxt = date(2026, m + 1, 1) if m < 12 else date(2027, 1, 1)
    return {
        "month": m,
        "month_pl": f"{MONTH_PL_LOC[m]} 2026",
        "month_en": f"{MONTH_EN[m]} 2026",
        "m_from": date(2026, m, 1).isoformat(),
        "m_to": nxt.isoformat(),
    }


def _shift_slots(kod: str) -> dict:
    return {"shift": kod, "shift_pl": SHIFT_PL[kod], "shift_en": SHIFT_EN[kod]}


def _desc(ctx: Ctx, column: str, code: str) -> dict:
    return {"pl": ctx.meaning(column, code, "pl"), "en": ctx.meaning(column, code, "en")}


# ---------------------------------------------------------------- samplers (test)

def s_line_shift(ctx, rng):
    lin, _, d, kod = rng.choice(ctx.prod_reports)
    return {"line": lin, **_date_slots(d), **_shift_slots(kod)}


def s_line_month(ctx, rng):
    return {"line": rng.choice(ctx.lines), **_month_slots(rng.randint(1, 3))}


def s_order_output(ctx, rng):
    return {"zlec": rng.choice(ctx.orders_with_output)}


def s_machine_month(ctx, rng):
    return {"mch": rng.choice(ctx.machines), **_month_slots(rng.randint(1, 3))}


STATUS_PL = {"R": "zleceń w realizacji", "W": "wstrzymanych zleceń",
             "N": "nowych, jeszcze nierozpoczętych zleceń", "Z": "zakończonych zleceń"}
STATUS_EN = {"R": "orders in progress", "W": "orders on hold",
             "N": "new orders that have not started yet", "Z": "completed orders"}


def s_status_group(ctx, rng):
    st, g = rng.choice("RWNZ"), rng.choice("ABCD")
    d = _desc(ctx, "PROD.GRUPA", g)
    return {"status": st, "status_pl": STATUS_PL[st], "status_en": STATUS_EN[st],
            "group": g, "group_pl": d["pl"], "group_en": d["en"]}


def s_line_week(ctx, rng):
    monday = date(2026, 1, 5) + timedelta(weeks=rng.randint(0, 11))
    sunday = monday + timedelta(days=6)
    return {"line": rng.choice(ctx.lines), **_date_slots(monday.isoformat(), "d1"),
            **_date_slots(sunday.isoformat(), "d2")}


def s_product(ctx, rng):
    return {"indeks": rng.choice(ctx.products)}


PRIO_PL = {"1": "wysokim priorytecie", "2": "normalnym priorytecie", "3": "niskim priorytecie"}
PRIO_EN = {"1": "high-priority", "2": "normal-priority", "3": "low-priority"}


def s_prio_month(ctx, rng):
    p = rng.choice("123")
    return {"prio": p, "prio_pl": PRIO_PL[p], "prio_en": PRIO_EN[p], **_month_slots(rng.randint(1, 4))}


# ---------------------------------------------------------------- samplers (train)

def s_machine_date(ctx, rng):
    _, mch, d, _ = rng.choice(ctx.prod_reports)
    return {"mch": mch, **_date_slots(d)}


def s_machine_any_date(ctx, rng):
    return {"mch": rng.choice(ctx.machines), **_date_slots(rng.choice(ctx.dates))}


def s_line_date(ctx, rng):
    return {"line": rng.choice(ctx.lines), **_date_slots(rng.choice(ctx.dates))}


def s_reason_month(ctx, rng):
    code = rng.choice(sorted(ctx.codes["PRZYCZ.PRZYCZ_KOD"]))
    d = _desc(ctx, "PRZYCZ.PRZYCZ_KOD", code)
    return {"reason": code, "reason_pl": d["pl"], "reason_en": d["en"], **_month_slots(rng.randint(1, 3))}


STATUS_PL_LIST = {"R": "zleceń w realizacji", "W": "wstrzymanych zleceń", "Z": "zakończonych zleceń",
                  "A": "anulowanych zleceń", "N": "nowych zleceń"}
STATUS_EN_LIST = {"R": "orders in progress", "W": "orders on hold", "Z": "completed orders",
                  "A": "cancelled orders", "N": "new orders"}


def s_product_status(ctx, rng):
    st = rng.choice("RWZAN")
    return {"indeks": rng.choice(ctx.products), "status": st,
            "status_pl": STATUS_PL_LIST[st], "status_en": STATUS_EN_LIST[st]}


def s_group_month(ctx, rng):
    g = rng.choice("ABCD")
    d = _desc(ctx, "PROD.GRUPA", g)
    return {"group": g, "group_pl": d["pl"], "group_en": d["en"], **_month_slots(rng.randint(1, 4))}


def s_crew_month(ctx, rng):
    return {"crew": rng.choice("ABCD"), **_month_slots(rng.randint(1, 3))}


KWAL_PL = {"1": "przyuczony", "2": "wykwalifikowany", "3": "starszy / ustawiacz"}
KWAL_EN = {"1": "trainee", "2": "qualified", "3": "senior / setter"}


def s_qual_line(ctx, rng):
    k = rng.choice("123")
    return {"line": rng.choice(ctx.lines), "kwal": k, "kwal_pl": KWAL_PL[k], "kwal_en": KWAL_EN[k]}


def s_group_prodstat(ctx, rng):
    g, st = rng.choice("ABCD"), rng.choice("AW")
    d = _desc(ctx, "PROD.GRUPA", g)
    return {"group": g, "group_pl": d["pl"], "group_en": d["en"], "pstat": st,
            "pstat_pl": {"A": "aktywnych", "W": "wycofanych"}[st],
            "pstat_en": {"A": "active", "W": "phased-out"}[st]}


def s_scrapcat_line_month(ctx, rng):
    c = rng.choice("MPWL")
    d = _desc(ctx, "BR_TYP.KAT", c)
    return {"cat": c, "cat_pl": d["pl"], "cat_en": d["en"], "line": rng.choice(ctx.lines),
            **_month_slots(rng.randint(1, 3))}


def s_due_date(ctx, rng):
    d = date(2026, 2, 1) + timedelta(days=rng.randint(0, 80))
    return _date_slots(d.isoformat())


def s_order_any(ctx, rng):
    return {"zlec": rng.choice(ctx.orders)}


def s_machine_ts(ctx, rng):
    h = rng.choice([8, 12, 16, 20])
    return {"mch": rng.choice(ctx.machines), **_date_slots(rng.choice(ctx.dates[1:])), "hour": f"{h:02d}"}


# ---------------------------------------------------------------- SQL building blocks

J_REJ_LIN = ("PROD_REJ r JOIN MCH m ON m.MCH_ID = r.MCH_ID JOIN LIN l ON l.LIN_ID = m.LIN_ID "
             "JOIN ZMIANA z ON z.ZM_ID = r.ZM_ID")
J_PRZ_LIN = ("PRZEST s JOIN MCH m ON m.MCH_ID = s.MCH_ID JOIN LIN l ON l.LIN_ID = m.LIN_ID "
             "JOIN ZMIANA z ON z.ZM_ID = s.ZM_ID")
MONTH_Z = "z.ZM_DATA >= '{m_from}' AND z.ZM_DATA < '{m_to}'"

TEMPLATES: list[Template] = [
    # ============================ TEST templates ============================
    Template(
        "T01", "test", "line output on a shift", s_line_shift,
        pl=("Ile dobrych sztuk wyprodukowała linia {line} na zmianie {shift_pl} {date_pl}?",
            "Jaka była dobra produkcja linii {line} {date_pl} na zmianie {shift_pl}?"),
        en=("How many good units did line {line} produce on the {shift_en} shift on {date_en}?",
            "What was the good output of line {line} on {date_en} during the {shift_en} shift?"),
        sql=f"SELECT SUM(r.ILOSC_OK) FROM {J_REJ_LIN} WHERE l.LIN_KOD = '{{line}}' "
            "AND z.ZM_DATA = '{date}' AND z.ZM_KOD = '{shift}'",
    ),
    Template(
        "T02", "test", "longest downtimes on a line", s_line_month,
        pl=("Pokaż 3 najdłuższe przestoje na linii {line} w {month_pl}: kod maszyny, opis przyczyny i czas w minutach.",
            "Które 3 przestoje na linii {line} w {month_pl} trwały najdłużej? Podaj kod maszyny, opis przyczyny i czas trwania w minutach."),
        en=("Show the 3 longest downtimes on line {line} in {month_en}: machine code, reason description and duration in minutes.",
            "Which 3 downtimes on line {line} in {month_en} lasted the longest? Give the machine code, reason description and duration in minutes."),
        sql=f"SELECT m.MCH_KOD, p.OPIS, s.CZAS_MIN FROM {J_PRZ_LIN} "
            "JOIN PRZYCZ p ON p.PRZYCZ_KOD = s.PRZYCZ_KOD "
            f"WHERE l.LIN_KOD = '{{line}}' AND {MONTH_Z} ORDER BY s.CZAS_MIN DESC LIMIT 3",
        order_check=(2, 3),
    ),
    Template(
        "T03", "test", "scrap rate of an order", s_order_output,
        pl=("Jaki jest procent braków dla zlecenia {zlec}?",
            "Ile procent produkcji zlecenia {zlec} stanowiły braki?"),
        en=("What is the scrap rate in percent for order {zlec}?",
            "What percentage of the output of order {zlec} was scrap?"),
        sql="SELECT 100.0 * SUM(r.ILOSC_BR) / (SUM(r.ILOSC_OK) + SUM(r.ILOSC_BR)) "
            "FROM PROD_REJ r JOIN ZLEC z ON z.ZLEC_ID = r.ZLEC_ID WHERE z.ZLEC_NR = '{zlec}'",
    ),
    Template(
        "T04", "test", "breakdowns of a machine in a month", s_machine_month,
        pl=("Ile awarii miała maszyna {mch} w {month_pl}?",
            "Ile razy maszyna {mch} uległa awarii w {month_pl}?"),
        en=("How many breakdowns did machine {mch} have in {month_en}?",
            "How many times did machine {mch} break down in {month_en}?"),
        sql="SELECT COUNT(*) FROM PRZEST s JOIN MCH m ON m.MCH_ID = s.MCH_ID "
            "JOIN PRZYCZ p ON p.PRZYCZ_KOD = s.PRZYCZ_KOD JOIN ZMIANA z ON z.ZM_ID = s.ZM_ID "
            f"WHERE m.MCH_KOD = '{{mch}}' AND p.KAT = 'T' AND {MONTH_Z}",
    ),
    Template(
        "T05", "test", "orders by status for a product group", s_status_group,
        pl=("Ile jest {status_pl} dla wyrobów z grupy „{group_pl}”?",
            "Podaj liczbę {status_pl} dotyczących grupy wyrobów „{group_pl}”."),
        en=("How many {status_en} are there for products in the '{group_en}' group?",
            "Give the number of {status_en} for the '{group_en}' product group."),
        sql="SELECT COUNT(*) FROM ZLEC z JOIN PROD p ON p.PROD_ID = z.PROD_ID "
            "WHERE z.ZLEC_STAT = '{status}' AND p.GRUPA = '{group}'",
    ),
    Template(
        "T06", "test", "unplanned downtime minutes of a line in a week", s_line_week,
        pl=("Ile minut łącznie trwały nieplanowane przestoje na linii {line} od {d1_pl} do {d2_pl}?",
            "Jaki był łączny czas nieplanowanych przestojów (w minutach) na linii {line} w dniach {d1_pl} – {d2_pl}?"),
        en=("What was the total duration in minutes of unplanned downtimes on line {line} from {d1_en} to {d2_en}?",
            "How many minutes of unplanned downtime did line {line} have between {d1_en} and {d2_en}?"),
        sql=f"SELECT SUM(s.CZAS_MIN) FROM {J_PRZ_LIN} WHERE l.LIN_KOD = '{{line}}' "
            "AND s.PLAN_FL = 'N' AND z.ZM_DATA BETWEEN '{d1}' AND '{d2}'",
    ),
    Template(
        "T07", "test", "top operator of a line in a month", s_line_month,
        pl=("Który operator wyprodukował najwięcej dobrych sztuk na linii {line} w {month_pl}? Podaj imię, nazwisko i tę liczbę.",
            "Kto z operatorów miał największą dobrą produkcję na linii {line} w {month_pl}? Podaj imię, nazwisko i ilość."),
        en=("Which operator produced the most good units on line {line} in {month_en}? Give first name, last name and the quantity.",
            "Who was the operator with the highest good output on line {line} in {month_en}? Give first name, last name and the quantity."),
        sql="SELECT o.IMIE, o.NAZW, SUM(r.ILOSC_OK) AS ILOSC FROM PROD_REJ r "
            "JOIN OPER o ON o.OPER_ID = r.OPER_ID JOIN MCH m ON m.MCH_ID = r.MCH_ID "
            "JOIN LIN l ON l.LIN_ID = m.LIN_ID JOIN ZMIANA z ON z.ZM_ID = r.ZM_ID "
            f"WHERE l.LIN_KOD = '{{line}}' AND {MONTH_Z} GROUP BY o.OPER_ID ORDER BY ILOSC DESC LIMIT 1",
        order_check=(2, 1),
    ),
    Template(
        "T08", "test", "top confirmed scrap types of a product", s_product,
        pl=("Jakie 3 typy braków miały największą ilość potwierdzonych braków dla wyrobu {indeks}? Podaj opis typu i ilość.",
            "Wymień 3 typy braków o największej potwierdzonej ilości dla wyrobu {indeks} wraz z opisem typu i ilością."),
        en=("Which 3 scrap types had the largest confirmed scrap quantity for product {indeks}? Give the type description and the quantity.",
            "List the 3 scrap types with the highest confirmed scrap quantity for product {indeks}, with type description and quantity."),
        sql="SELECT t.OPIS, SUM(b.ILOSC) AS ILOSC FROM BRAKI b "
            "JOIN BR_TYP t ON t.BR_TYP_KOD = b.BR_TYP_KOD JOIN PROD_REJ r ON r.REJ_ID = b.REJ_ID "
            "JOIN ZLEC z ON z.ZLEC_ID = r.ZLEC_ID JOIN PROD p ON p.PROD_ID = z.PROD_ID "
            "WHERE p.INDEKS = '{indeks}' AND b.BR_STAT = 'P' "
            "GROUP BY t.BR_TYP_KOD ORDER BY ILOSC DESC LIMIT 3",
        order_check=(1, 3),
    ),
    Template(
        "T09", "test", "average output per shift by crew", s_line_month,
        pl=("Jaka była średnia dobra produkcja na jedną zmianę dla każdej brygady na linii {line} w {month_pl}?",
            "Ile średnio dobrych sztuk na zmianę produkowała każda brygada na linii {line} w {month_pl}?"),
        en=("What was the average good output per shift for each crew on line {line} in {month_en}?",
            "On average, how many good units per shift did each crew produce on line {line} in {month_en}?"),
        sql="SELECT t.BRYG, AVG(t.ILOSC) FROM (SELECT z.BRYG, z.ZM_ID, SUM(r.ILOSC_OK) AS ILOSC "
            f"FROM {J_REJ_LIN} WHERE l.LIN_KOD = '{{line}}' AND {MONTH_Z} "
            "GROUP BY z.ZM_ID, z.BRYG) AS t GROUP BY t.BRYG",
    ),
    Template(
        "T10", "test", "cancelled orders by priority and planned start", s_prio_month,
        pl=("Ile anulowanych zleceń o {prio_pl} miało planowany start w {month_pl}?",
            "Podaj liczbę anulowanych zleceń o {prio_pl} z planowaną datą rozpoczęcia w {month_pl}."),
        en=("How many cancelled {prio_en} orders had a planned start in {month_en}?",
            "Give the number of cancelled {prio_en} orders with a planned start date in {month_en}."),
        sql="SELECT COUNT(*) FROM ZLEC WHERE ZLEC_STAT = 'A' AND PRIOR = '{prio}' "
            "AND DT_PLAN_OD >= '{m_from}' AND DT_PLAN_OD < '{m_to}'",
    ),
    # ============================ TRAIN templates ===========================
    Template(
        "R01", "train", "machine output on a date", s_machine_date,
        pl=("Ile dobrych sztuk wyprodukowała maszyna {mch} {date_pl}?",
            "Jaka była dobra produkcja maszyny {mch} w dniu {date_pl}?"),
        en=("How many good units did machine {mch} produce on {date_en}?",
            "What was the good output of machine {mch} on {date_en}?"),
        sql="SELECT SUM(r.ILOSC_OK) FROM PROD_REJ r JOIN MCH m ON m.MCH_ID = r.MCH_ID "
            "JOIN ZMIANA z ON z.ZM_ID = r.ZM_ID WHERE m.MCH_KOD = '{mch}' AND z.ZM_DATA = '{date}'",
    ),
    Template(
        "R02", "train", "number of downtimes of a machine on a date", s_machine_any_date,
        pl=("Ile przestojów zarejestrowano na maszynie {mch} {date_pl}?",
            "Ile razy maszyna {mch} stała {date_pl}?"),
        en=("How many downtimes were recorded on machine {mch} on {date_en}?",
            "How many times was machine {mch} down on {date_en}?"),
        sql="SELECT COUNT(*) FROM PRZEST s JOIN MCH m ON m.MCH_ID = s.MCH_ID "
            "JOIN ZMIANA z ON z.ZM_ID = s.ZM_ID WHERE m.MCH_KOD = '{mch}' AND z.ZM_DATA = '{date}'",
    ),
    Template(
        "R03", "train", "number of shifts a machine produced in a month", s_machine_month,
        pl=("Na ilu zmianach maszyna {mch} raportowała produkcję w {month_pl}?",
            "Przez ile zmian maszyna {mch} produkowała w {month_pl}?"),
        en=("On how many shifts did machine {mch} report production in {month_en}?",
            "For how many shifts was machine {mch} producing in {month_en}?"),
        sql="SELECT COUNT(DISTINCT r.ZM_ID) FROM PROD_REJ r JOIN MCH m ON m.MCH_ID = r.MCH_ID "
            f"JOIN ZMIANA z ON z.ZM_ID = r.ZM_ID WHERE m.MCH_KOD = '{{mch}}' AND {MONTH_Z}",
    ),
    Template(
        "R04", "train", "planned downtime minutes of a line in a month", s_line_month,
        pl=("Ile minut planowanych przestojów miała linia {line} w {month_pl}?",
            "Jaki był łączny czas przestojów planowanych na linii {line} w {month_pl}?"),
        en=("How many minutes of planned downtime did line {line} have in {month_en}?",
            "What was the total planned downtime on line {line} in {month_en}?"),
        sql=f"SELECT SUM(s.CZAS_MIN) FROM {J_PRZ_LIN} WHERE l.LIN_KOD = '{{line}}' "
            f"AND s.PLAN_FL = 'T' AND {MONTH_Z}",
    ),
    Template(
        "R05", "train", "machine with most breakdown minutes on a line", s_line_month,
        pl=("Która maszyna na linii {line} miała najdłuższy łączny czas awarii w {month_pl}? Podaj kod i minuty.",
            "Podaj kod maszyny z linii {line} o największej liczbie minut awarii w {month_pl} i tę liczbę."),
        en=("Which machine on line {line} had the longest total breakdown time in {month_en}? Give the code and minutes.",
            "Give the code of the machine on line {line} with the most breakdown minutes in {month_en}, and the minutes."),
        sql=f"SELECT m.MCH_KOD, SUM(s.CZAS_MIN) AS MIN_AW FROM {J_PRZ_LIN} "
            "JOIN PRZYCZ p ON p.PRZYCZ_KOD = s.PRZYCZ_KOD "
            f"WHERE l.LIN_KOD = '{{line}}' AND p.KAT = 'T' AND {MONTH_Z} "
            "GROUP BY m.MCH_ID ORDER BY MIN_AW DESC LIMIT 1",
        order_check=(1, 1),
    ),
    Template(
        "R06", "train", "average downtime length for a reason", s_reason_month,
        pl=("Jaki był średni czas przestoju (w minutach) z przyczyny „{reason_pl}” w {month_pl}?",
            "Ile średnio minut trwał przestój z powodu „{reason_pl}” w {month_pl}?"),
        en=("What was the average downtime duration in minutes for the reason '{reason_en}' in {month_en}?",
            "On average, how many minutes did a downtime caused by '{reason_en}' last in {month_en}?"),
        sql="SELECT AVG(s.CZAS_MIN) FROM PRZEST s JOIN ZMIANA z ON z.ZM_ID = s.ZM_ID "
            f"WHERE s.PRZYCZ_KOD = '{{reason}}' AND {MONTH_Z}",
    ),
    Template(
        "R07", "train", "order numbers for a product by status", s_product_status,
        pl=("Wypisz numery {status_pl} dla wyrobu {indeks}.",
            "Jakie są numery {status_pl} na wyrób {indeks}?"),
        en=("List the numbers of {status_en} for product {indeks}.",
            "What are the order numbers of {status_en} for product {indeks}?"),
        sql="SELECT z.ZLEC_NR FROM ZLEC z JOIN PROD p ON p.PROD_ID = z.PROD_ID "
            "WHERE p.INDEKS = '{indeks}' AND z.ZLEC_STAT = '{status}'",
    ),
    Template(
        "R08", "train", "ordered quantity for a product group by planned start month", s_group_month,
        pl=("Jaka jest łączna zlecona ilość w zleceniach na wyroby z grupy „{group_pl}” z planowanym startem w {month_pl}?",
            "Ile sztuk łącznie zlecono dla grupy „{group_pl}” w zleceniach planowanych do rozpoczęcia w {month_pl}?"),
        en=("What is the total ordered quantity in orders for '{group_en}' products with a planned start in {month_en}?",
            "How many units in total were ordered for the '{group_en}' group in orders planned to start in {month_en}?"),
        sql="SELECT SUM(z.ZLEC_ILOSC) FROM ZLEC z JOIN PROD p ON p.PROD_ID = z.PROD_ID "
            "WHERE p.GRUPA = '{group}' AND z.DT_PLAN_OD >= '{m_from}' AND z.DT_PLAN_OD < '{m_to}'",
    ),
    Template(
        "R09", "train", "shifts worked by a crew in a month", s_crew_month,
        pl=("Ile zmian przepracowała brygada {crew} w {month_pl}?",
            "Na ilu zmianach pracowała brygada {crew} w {month_pl}?"),
        en=("How many shifts did crew {crew} work in {month_en}?",
            "On how many shifts was crew {crew} working in {month_en}?"),
        sql="SELECT COUNT(*) FROM ZMIANA WHERE BRYG = '{crew}' "
            "AND ZM_DATA >= '{m_from}' AND ZM_DATA < '{m_to}'",
    ),
    Template(
        "R10", "train", "active operators of a qualification on a line", s_qual_line,
        pl=("Ilu aktywnych operatorów z kwalifikacją „{kwal_pl}” jest przypisanych do linii {line}?",
            "Podaj liczbę aktywnych operatorów o kwalifikacji „{kwal_pl}” na linii {line}."),
        en=("How many active operators with the '{kwal_en}' qualification are assigned to line {line}?",
            "Give the number of active operators with the '{kwal_en}' qualification on line {line}."),
        sql="SELECT COUNT(*) FROM OPER o JOIN LIN l ON l.LIN_ID = o.LIN_ID "
            "WHERE l.LIN_KOD = '{line}' AND o.OPER_STAT = 'A' AND o.KWAL = '{kwal}'",
    ),
    Template(
        "R11", "train", "product indexes in a group by status", s_group_prodstat,
        pl=("Wypisz indeksy {pstat_pl} wyrobów z grupy „{group_pl}”.",
            "Jakie indeksy mają {pstat_pl} wyroby z grupy „{group_pl}”?"),
        en=("List the indexes of {pstat_en} products in the '{group_en}' group.",
            "What are the indexes of {pstat_en} products in the '{group_en}' group?"),
        sql="SELECT INDEKS FROM PROD WHERE GRUPA = '{group}' AND PROD_STAT = '{pstat}'",
    ),
    Template(
        "R12", "train", "confirmed scrap of a defect category on a line", s_scrapcat_line_month,
        pl=("Ile sztuk potwierdzonych braków z kategorii „{cat_pl}” zgłoszono na linii {line} w {month_pl}?",
            "Jaka była ilość potwierdzonych braków typu „{cat_pl}” na linii {line} w {month_pl}?"),
        en=("How many units of confirmed scrap in the '{cat_en}' category were reported on line {line} in {month_en}?",
            "What was the confirmed scrap quantity of the '{cat_en}' category on line {line} in {month_en}?"),
        sql="SELECT SUM(b.ILOSC) FROM BRAKI b JOIN BR_TYP t ON t.BR_TYP_KOD = b.BR_TYP_KOD "
            "JOIN PROD_REJ r ON r.REJ_ID = b.REJ_ID JOIN MCH m ON m.MCH_ID = r.MCH_ID "
            "JOIN LIN l ON l.LIN_ID = m.LIN_ID JOIN ZMIANA z ON z.ZM_ID = r.ZM_ID "
            "WHERE t.KAT = '{cat}' AND b.BR_STAT = 'P' "
            f"AND l.LIN_KOD = '{{line}}' AND {MONTH_Z}",
    ),
    Template(
        "R13", "train", "machines of a line with a breakdown on a date", s_line_date,
        pl=("Które maszyny na linii {line} miały awarię {date_pl}?",
            "Podaj kody maszyn z linii {line}, na których {date_pl} wystąpiła awaria."),
        en=("Which machines on line {line} had a breakdown on {date_en}?",
            "Give the codes of machines on line {line} that broke down on {date_en}."),
        sql=f"SELECT DISTINCT m.MCH_KOD FROM {J_PRZ_LIN} JOIN PRZYCZ p ON p.PRZYCZ_KOD = s.PRZYCZ_KOD "
            "WHERE l.LIN_KOD = '{line}' AND p.KAT = 'T' AND z.ZM_DATA = '{date}'",
    ),
    Template(
        "R14", "train", "operators who worked on a machine on a date", s_machine_date,
        pl=("Kto pracował na maszynie {mch} {date_pl}? Podaj imię i nazwisko.",
            "Którzy operatorzy obsługiwali maszynę {mch} w dniu {date_pl}? Podaj imiona i nazwiska."),
        en=("Who worked on machine {mch} on {date_en}? Give first and last name.",
            "Which operators ran machine {mch} on {date_en}? Give first and last names."),
        sql="SELECT DISTINCT o.IMIE, o.NAZW FROM OPER_ZM oz JOIN OPER o ON o.OPER_ID = oz.OPER_ID "
            "JOIN MCH m ON m.MCH_ID = oz.MCH_ID JOIN ZMIANA z ON z.ZM_ID = oz.ZM_ID "
            "WHERE m.MCH_KOD = '{mch}' AND z.ZM_DATA = '{date}'",
    ),
    Template(
        "R15", "train", "open high-priority orders due before a date", s_due_date,
        pl=("Wypisz numery niezakończonych zleceń o wysokim priorytecie z planowanym końcem przed {date_pl}.",
            "Które otwarte zlecenia o wysokim priorytecie miały być zakończone przed {date_pl}? Podaj numery."),
        en=("List the numbers of unfinished high-priority orders with a planned end before {date_en}.",
            "Which open high-priority orders were planned to finish before {date_en}? Give their numbers."),
        sql="SELECT ZLEC_NR FROM ZLEC WHERE PRIOR = '1' AND ZLEC_STAT IN ('N', 'R', 'W') "
            "AND DT_PLAN_DO < '{date}'",
    ),
    Template(
        "R16", "train", "number of operations in an order", s_order_any,
        pl=("Z ilu operacji składa się zlecenie {zlec}?",
            "Ile pozycji (operacji) ma zlecenie {zlec}?"),
        en=("How many operations does order {zlec} consist of?",
            "How many positions (operations) does order {zlec} have?"),
        sql="SELECT COUNT(*) FROM ZLEC_POZ zp JOIN ZLEC z ON z.ZLEC_ID = zp.ZLEC_ID "
            "WHERE z.ZLEC_NR = '{zlec}'",
    ),
    Template(
        "R17", "train", "best shift of a line in a month", s_line_month,
        pl=("Na której zmianie linia {line} wyprodukowała najwięcej dobrych sztuk w {month_pl}? Podaj datę, numer zmiany i ilość.",
            "Która zmiana (data i numer) była najlepsza pod względem dobrej produkcji na linii {line} w {month_pl}? Podaj też ilość."),
        en=("On which shift did line {line} produce the most good units in {month_en}? Give the date, shift number and quantity.",
            "Which shift (date and number) had the highest good output on line {line} in {month_en}? Also give the quantity."),
        sql=f"SELECT z.ZM_DATA, z.ZM_KOD, SUM(r.ILOSC_OK) AS ILOSC FROM {J_REJ_LIN} "
            f"WHERE l.LIN_KOD = '{{line}}' AND {MONTH_Z} GROUP BY z.ZM_ID ORDER BY ILOSC DESC LIMIT 1",
        order_check=(2, 1),
    ),
    Template(
        "R18", "train", "scrap reports awaiting confirmation for a machine", s_machine_month,
        pl=("Ile zgłoszeń braków z maszyny {mch} czeka na potwierdzenie z {month_pl}?",
            "Ile niepotwierdzonych jeszcze zgłoszeń braków dotyczy maszyny {mch} w {month_pl}?"),
        en=("How many scrap reports from machine {mch} in {month_en} are still awaiting confirmation?",
            "How many not yet confirmed scrap reports concern machine {mch} in {month_en}?"),
        sql="SELECT COUNT(*) FROM BRAKI b JOIN PROD_REJ r ON r.REJ_ID = b.REJ_ID "
            "JOIN MCH m ON m.MCH_ID = r.MCH_ID JOIN ZMIANA z ON z.ZM_ID = r.ZM_ID "
            f"WHERE m.MCH_KOD = '{{mch}}' AND b.BR_STAT = 'Z' AND {MONTH_Z}",
    ),
    Template(
        "R19", "train", "longest single downtime of a machine in a month", s_machine_month,
        pl=("Ile minut trwał najdłuższy pojedynczy przestój maszyny {mch} w {month_pl}?",
            "Jaki był najdłuższy przestój maszyny {mch} w {month_pl} (w minutach)?"),
        en=("How many minutes did the longest single downtime of machine {mch} last in {month_en}?",
            "What was the longest downtime of machine {mch} in {month_en}, in minutes?"),
        sql="SELECT MAX(s.CZAS_MIN) FROM PRZEST s JOIN MCH m ON m.MCH_ID = s.MCH_ID "
            f"JOIN ZMIANA z ON z.ZM_ID = s.ZM_ID WHERE m.MCH_KOD = '{{mch}}' AND {MONTH_Z}",
    ),
    Template(
        "R20", "train", "last machine status before a moment", s_machine_ts,
        pl=("Jaki był ostatni zarejestrowany status maszyny {mch} przed {date_pl}, godz. {hour}:00? Podaj czas zmiany i kod statusu.",
            "Podaj czas i kod ostatniej zmiany statusu maszyny {mch} przed {date_pl} {hour}:00."),
        en=("What was the last recorded status of machine {mch} before {date_en} {hour}:00? Give the change time and status code.",
            "Give the time and code of the last status change of machine {mch} before {date_en} {hour}:00."),
        sql="SELECT ms.TS, ms.STAT FROM MCH_STAT ms JOIN MCH m ON m.MCH_ID = ms.MCH_ID "
            "WHERE m.MCH_KOD = '{mch}' AND ms.TS < '{date} {hour}:00' ORDER BY ms.TS DESC LIMIT 1",
        order_check=(0, 1),
    ),
]

TEMPLATES_BY_ID = {t.id: t for t in TEMPLATES}
