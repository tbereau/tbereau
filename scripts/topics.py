"""Join the hand-maintained topic map onto the exported publication data.

`research_topics.qmd` selects the papers for each of its sections. It used to
do that either by coauthor or by a list of DOIs written into the page, and both
drift out of date silently -- see the header of `data/topics.yml`. The
assignment now lives in that file, and this module attaches it to
`data/all_papers.{json,csv}` as a `topics` field so the page can filter on
`p.topics.includes("gen-free-energy")`.

The two checks in `report_gaps` are the point of the exercise as much as the
join is: an unknown DOI means a typo that would have quietly emptied a section,
and an unassigned paper means a publication that appears nowhere on the page.
Neither is visible in the rendered site, which is how the previous filters
managed to hide half the record.

Run standalone after editing the map::

    poetry run python scripts/topics.py

`fetch_papers.py` and `embed_papers.py` both write the exports through
`write_exports` here, since either one overwrites what the other produced.
"""

import csv
import json
import sys
from pathlib import Path

import yaml

try:
    from to_quarto.utils import ROOT_DIR
except ImportError:  # invoked as `python scripts/topics.py` from the repo root
    ROOT_DIR = Path(__file__).parent.parent.absolute()

TOPICS_FILE = Path(ROOT_DIR).joinpath("data", "topics.yml")
PAPERS_JSON = Path(ROOT_DIR).joinpath("data", "all_papers.json")
PAPERS_CSV = Path(ROOT_DIR).joinpath("data", "all_papers.csv")


# The dataset is inconsistent about DOI case -- `10.1039/D5SC03855C` alongside
# `10.1103/physrevlett.121.256002` -- so every lookup goes through this.
def _key(doi):
    return str(doi).strip().lower()


def load_topic_map(path=TOPICS_FILE):
    """Return {doi: [topic key, ...]}, preserving the order topics appear in."""
    spec = yaml.safe_load(Path(path).read_text())
    topic_map = {}
    for topic in spec["topics"]:
        for doi in topic["dois"]:
            topic_map.setdefault(_key(doi), []).append(topic["key"])
    return topic_map


def topics_for(dois, topic_map):
    return [topic_map.get(_key(doi), []) for doi in dois]


def report_gaps(dois, topic_map):
    """Warn about DOIs the map invents and papers it forgets. Returns a count."""
    known = {_key(doi) for doi in dois}
    unknown = sorted(doi for doi in topic_map if doi not in known)
    unassigned = sorted(doi for doi in known if not topic_map.get(doi))

    if unknown:
        print(
            f"WARNING: {len(unknown)} DOI(s) in {TOPICS_FILE.name} match no "
            "publication, so their section is short by that many rows:"
        )
        for doi in unknown:
            print(f"  {doi}")
    if unassigned:
        print(
            f"WARNING: {len(unassigned)} publication(s) belong to no topic and "
            "so appear nowhere on the research-topics page:"
        )
        for doi in unassigned:
            print(f"  {doi}")
    return len(unknown) + len(unassigned)


def add_topics_column(df, path=TOPICS_FILE):
    """Attach the `topics` column to a papers DataFrame, in place."""
    topic_map = load_topic_map(path)
    report_gaps(df["doi"].tolist(), topic_map)
    df["topics"] = topics_for(df["doi"].tolist(), topic_map)
    return df


def write_exports(df, path=TOPICS_FILE):
    """Tag a papers DataFrame and write both exports `data/` serves to the site.

    The two formats need different representations of a list-valued column --
    JSON takes the list, CSV would otherwise get its Python repr -- so both
    writes live here rather than at each call site.
    """
    add_topics_column(df, path)
    df.to_json(PAPERS_JSON)
    df = df.copy()
    df["topics"] = df["topics"].map(";".join)
    df.to_csv(PAPERS_CSV)


def _rewrite_json(topic_map):
    data = json.loads(PAPERS_JSON.read_text())
    dois = data["doi"]  # {row index as a string: doi}, pandas' `orient="columns"`
    gaps = report_gaps(list(dois.values()), topic_map)
    data["topics"] = {i: topic_map.get(_key(doi), []) for i, doi in dois.items()}
    PAPERS_JSON.write_text(json.dumps(data))
    tagged = sum(1 for topics in data["topics"].values() if topics)
    return tagged, len(dois), gaps


def _rewrite_csv(topic_map):
    with PAPERS_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    fieldnames = [f for f in rows[0] if f != "topics"] + ["topics"]
    for row in rows:
        # Joined rather than written as a list, so that the landscape page --
        # which reads this file with `.csv({typed: true})` -- gets a string
        # rather than a stringified Python repr.
        row["topics"] = ";".join(topic_map.get(_key(row["doi"]), []))
    with PAPERS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    topic_map = load_topic_map()
    tagged, n_papers, gaps = _rewrite_json(topic_map)
    _rewrite_csv(topic_map)
    n_topics = len({key for keys in topic_map.values() for key in keys})
    print(f"tagged {tagged} of {n_papers} publications across {n_topics} topics")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
