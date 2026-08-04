import logging

from my_scientific_profile.papers.papers import Paper, VenueKind

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "_Not available._"

# How each provider is credited on the page. The library reports which one
# supplied each field, so these footers follow the data instead of asserting a
# fixed pipeline that no longer holds.
PROVIDER_NAMES = {
    "crossref": "CrossRef",
    "openalex": "OpenAlex",
    "datacite": "DataCite",
    "dblp": "DBLP",
    "semantic_scholar": "Semantic Scholar",
    "unpaywall": "Unpaywall",
    "doi2bib": "doi2bib",
    "orcid": "ORCID",
    "config": "manual entry",
    "rendered": "this site",
}

# Altmetric tracks Crossref DOIs; an arXiv or Zenodo DOI renders an empty badge.
UNTRACKED_DOI_PREFIXES = ("10.48550/", "10.5281/")

BROWSER_ICON = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
        class="bi bi-browser-chrome" viewBox="0 0 16 16">
    <path fill-rule="evenodd"
       d="M16 8a8.001 8.001 0 0 1-7.022 7.94l1.902-7.098a2.995
       2.995 0 0 0 .05-1.492A2.977 2.977 0 0 0 10.237 6h5.511A8
       8 0 0 1 16 8ZM0 8a8 8 0 0 0 7.927 8l1.426-5.321a2.978
       2.978 0 0 1-.723.255 2.979 2.979 0 0 1-1.743-.147 2.986
       2.986 0 0 1-1.043-.7L.633 4.876A7.975 7.975 0 0 0 0
       8Zm5.004-.167L1.108 3.936A8.003 8.003 0 0 1 15.418
       5H8.066a2.979 2.979 0 0 0-1.252.243 2.987 2.987 0 0
       0-1.81 2.59ZM8 10a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"/>
    </svg>"""

PDF_ICON = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
        class="bi bi-file-pdf" viewBox="0 0 16 16">
    <path
        d="M4 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2
        2 0 0 0 2-2V2a2 2 0 0 0-2-2H4zm0 1h8a1 1 0
        0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1
        1 0 0 1 1-1z"/>
    <path
        d="M4.603 12.087a.81.81 0 0
        1-.438-.42c-.195-.388-.13-.776.08-1.102.198-.307.526-.568.897-.787a7.68
        7.68 0 0 1 1.482-.645 19.701 19.701 0 0 0 1.062-2.227 7.269 7.269 0 0
        1-.43-1.295c-.086-.4-.119-.796-.046-1.136.075-.354.274-.672.65-.823.192-.077.4-.12.602-.077a.7.7
        0 0 1 .477.365c.088.164.12.356.127.538.007.187-.012.395-.047.614-.084.51-.27
        1.134-.52 1.794a10.954 10.954 0 0 0 .98 1.686 5.753 5.753 0 0 1
        1.334.05c.364.065.734.195.96.465.12.144.193.32.2.518.007.192-.047.382-.138.563a1.04
        1.04 0 0 1-.354.416.856.856 0 0 1-.51.138c-.331-.014-.654-.196-.933-.417a5.716
        5.716 0 0 1-.911-.95 11.642 11.642 0 0 0-1.997.406 11.311 11.311 0 0 1-1.021
        1.51c-.29.35-.608.655-.926.787a.793.793 0 0
        1-.58.029zm1.379-1.901c-.166.076-.32.156-.459.238-.328.194-.541.383-.647.547-.094.145-.096.25-.04.361.01.022.02.036.026.044a.27.27
        0 0 0 .035-.012c.137-.056.355-.235.635-.572a8.18 8.18 0 0 0
        .45-.606zm1.64-1.33a12.647
        12.647 0 0 1 1.01-.193 11.666 11.666 0 0
        1-.51-.858 20.741 20.741 0 0 1-.5
        1.05zm2.446.45c.15.162.296.3.435.41.24.19.407.253.498.256a.107.107
        0 0 0 .07-.015.307.307 0 0 0 .094-.125.436.436 0 0 0
        .059-.2.095.095 0 0 0-.026-.063c-.052-.062-.2-.152-.518-.209a3.881 3.881 0 0
        0-.612-.053zM8.078 5.8a6.7 6.7 0 0 0
        .2-.828c.031-.188.043-.343.038-.465a.613.613 0
        0 0-.032-.198.517.517 0 0
        0-.145.04c-.087.035-.158.106-.196.283-.04.192-.03.469.046.822.024.111.054.227.09.346z"/>
    </svg>"""


def credit(paper: Paper, field: str) -> str:
    """A footer naming whichever provider actually supplied a field."""
    provider = paper.source_of(field)
    if not provider:
        return ""
    name = PROVIDER_NAMES.get(provider, provider)
    return (
        '```{=html}\n<footer class="blockquote-footer">from '
        f'<cite title="Source Title">{name}</cite>\n</footer>\n```'
    )


def altmetric_badge(paper: Paper) -> str:
    """The Altmetric donut, omitted where Altmetric has nothing to show."""
    if paper.doi.startswith(UNTRACKED_DOI_PREFIXES):
        return ""
    return (
        "```{=html}\n"
        '<div class="float-end">\n'
        "    <div data-badge-type='donut' class='altmetric-embed'\n"
        f"    data-badge-popover='left' data-doi=\"{paper.doi}\"></div>\n"
        "</div>\n"
        "<script type='text/javascript'\n"
        "  src='https://d1bxh8uas1mnw7.cloudfront.net/assets/embed.js'></script>\n"
        "<br><br><br>\n"
        "```"
    )


def open_access_button(url: str | None, icon: str, label: str) -> str:
    """One open-access link, or nothing at all when there is no URL to link to.

    A missing URL used to still render a disabled button, interpolating Python's
    `None` into the href so the markup pointed at `papers/None`. A button nobody
    can press tells the reader nothing, so leave it out instead.
    """
    if not url:
        return ""
    return (
        f'<a class="btn btn-outline-primary" href="{url}" role="button">\n'
        f"    {icon}\n"
        f"    {label}\n"
        "    </a>"
    )


def open_access_section(paper: Paper) -> str:
    """The Open Access tab, present only when Unpaywall found something.

    Unpaywall has no open record for roughly a third of these papers. An empty
    tab is a dead end, so the tab itself is conditional on there being a link.
    """
    buttons = [
        button
        for button in (
            open_access_button(
                paper.open_access.landing_page_url, BROWSER_ICON, "Webpage"
            ),
            open_access_button(paper.open_access.pdf_url, PDF_ICON, "PDF"),
        )
        if button
    ]
    if not buttons:
        return ""
    opening = (
        "\n## Open Access\n"
        "```{=html}\n"
        '<div class="float-end">\n'
        '  <img src="../static/open_access.png" height="64">\n'
        "</div>\n"
    )
    closing = f"\n    <p></p>\n```\n{credit(paper, 'open_access')}\n"
    return opening + "\n".join(buttons) + closing


def save_quarto_paper_page_to_file(paper: Paper, filename: str) -> None:
    logger.info(f"saving quarto author page for {paper.title} to {filename}")
    with open(filename, "w") as stream:
        stream.write(generate_quarto_paper_page(paper))


def format_description(paper: Paper) -> str:
    """The one-line venue citation shown in the listing.

    Only journal articles have a volume, so the shape has to vary: a conference
    paper or preprint would otherwise read "**None**".
    """
    venue = paper.journal
    parts = []
    if venue.abbreviation:
        parts.append(f"_{venue.abbreviation}_")
    if venue.kind == VenueKind.PREPRINT:
        parts.append("preprint")
    if venue.volume:
        parts.append(f"**{venue.volume}**")
    parts.append(f"({paper.year})")
    return " ".join(parts)


def generate_quarto_paper_page(paper: Paper) -> str:
    authors_list = [a.family for a in paper.authors]
    authors = ""
    for author in authors_list[:-1]:
        authors += author + ", "
    authors += authors_list[-1]
    abstract = paper.abstract or NOT_AVAILABLE
    tldr = paper.tldr or NOT_AVAILABLE
    bibtex = (
        "```{bibtex}\n" + paper.bib_entry + "\n```"
        if paper.bib_entry
        else NOT_AVAILABLE
    )
    altmetric = altmetric_badge(paper)
    open_access = open_access_section(paper)
    header = f"""
---
title: "{paper.title}"
doi: "{paper.doi}"
date: "{paper.publication_date}"
date-format: iso
authors: "{authors}"
description: "{format_description(paper)}"
year: {paper.year}
format:
  html:
    toc: false
    echo: false
    keep-hidden: true
    code-tools: false
---
"""
    body = f"""
{altmetric}

::: {{.panel-tabset}}

## Abstract

{abstract}
{credit(paper, "abstract")}

## TL;DR

{tldr}
{credit(paper, "tldr")}

## BibTeX

{bibtex}
{credit(paper, "bib_entry")}
{open_access}
:::
"""
    return header + body
