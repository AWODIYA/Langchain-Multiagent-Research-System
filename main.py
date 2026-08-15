from rich import print
from src.tools.tools import web_search,  search_academic_papers, scrape_url, read_pdf_from_url

# output = search_academic_papers.invoke("paged attention",
#                                  limit=5, year_range="2017-2025")
output = web_search.invoke("Attention is all you need paper")
# output = scrape_url.invoke("https://www.nature.com/articles/s41746-025-01460-1")
# output = read_pdf_from_url.invoke("https://arxiv.org/pdf/2506.07311.pdf")
print(output)


