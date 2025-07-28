import requests


def search_pubmed(query, max_results=3):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    search_resp = requests.get(search_url, params=search_params)
    ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return "No articles found."

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }
    fetch_resp = requests.get(fetch_url, params=fetch_params)
    return fetch_resp.text
