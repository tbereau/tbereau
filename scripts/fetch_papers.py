from dotenv import load_dotenv

load_dotenv()


from my_scientific_profile.papers.papers import (
    fetch_all_paper_authors,
    fetch_all_paper_infos,
    get_last_resolution_report,
)

papers = fetch_all_paper_infos()
report = get_last_resolution_report()
paper_authors = fetch_all_paper_authors()
print(f"fetched {len(papers)} papers and {len(paper_authors)} authors")
if report:
    print(f"{len(report)} works needed attention:")
    for issue in report:
        print(f"  {issue.key}: [{issue.stage}] {issue.message}")

# Nothing else would ever mention a paper that is missing from the ORCID record,
# which is how one goes unnoticed for months.
from my_scientific_profile.discovery import find_unlisted_works  # noqa: E402

unlisted = find_unlisted_works()
if unlisted:
    print(f"\n{len(unlisted)} works are attributed to your ORCID but not listed:")
    for work in unlisted:
        print(f"  {work}")
    print("  Add them on ORCID, or set `ignore: true` in the config to stop asking.")

from my_scientific_profile.database.authors import save_all_paper_authors_to_s3  # noqa
from my_scientific_profile.database.papers import save_all_papers_to_s3, convert_papers_to_dataframe  # noqa
from my_scientific_profile.database.aws_s3 import S3_BUCKET, S3_CLIENT  # noqa

save_all_papers_to_s3(s3_client=S3_CLIENT, s3_bucket=S3_BUCKET, papers=papers)
save_all_paper_authors_to_s3(s3_client=S3_CLIENT, s3_bucket=S3_BUCKET)
print(f"saved to S3 {S3_CLIENT}")

df = convert_papers_to_dataframe(papers)

# Tags each paper with the section of the research-topics page it belongs to,
# and warns about anything newly fetched that has not been assigned one yet.
from topics import write_exports  # noqa: E402

write_exports(df)

# The homepage's recent-work list is generated rather than read in the browser,
# so it has to be rebuilt whenever the exports change.
from recent_work import write_include  # noqa: E402

write_include()
