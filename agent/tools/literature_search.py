"""PubMed/PMC retrieval for the evidence-grounded diagnosis flow.

The adapter stores retrieved source content, not an LLM-generated summary. This
keeps the later evidence extraction auditable and makes every claim traceable
to a PMID/PMCID/DOI and the original abstract or full text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from .candidate_metadata import canonical_omim_id
from ..state.state_types import MergedDiseaseCandidateV2, SourceRecord


NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
DEFAULT_MAX_RESULTS = int(os.getenv("PUBMED_MAX_RESULTS_PER_CANDIDATE", "2"))
DEFAULT_MAX_QUERIES = max(4, int(os.getenv("PUBMED_MAX_QUERIES_PER_CANDIDATE", "6")))
LITERATURE_MAX_WORKERS = max(1, int(os.getenv("PUBMED_MAX_WORKERS", "2")))
REQUEST_TIMEOUT = float(os.getenv("PUBMED_REQUEST_TIMEOUT_SECONDS", "20"))
REQUEST_RETRIES = max(0, int(os.getenv("LITERATURE_REQUEST_RETRIES", "1")))
MAX_ABSTRACT_LENGTH = int(os.getenv("PUBMED_MAX_ABSTRACT_LENGTH", "12000"))
MAX_FULL_TEXT_LENGTH = int(os.getenv("PMC_MAX_FULL_TEXT_LENGTH", "50000"))
NCBI_TOOL = os.getenv("NCBI_TOOL", "AI_AgentWithLangGraph")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_MAX_REQUESTS_PER_SECOND = max(
    1.0, float(os.getenv("NCBI_MAX_REQUESTS_PER_SECOND", "3"))
)
EUROPE_PMC_MAX_RESULTS = max(
    1, int(os.getenv("EUROPE_PMC_MAX_RESULTS_PER_CANDIDATE", str(DEFAULT_MAX_RESULTS)))
)
EUROPE_PMC_ALLOWED_SOURCES = {
    value.strip().upper()
    for value in os.getenv("EUROPE_PMC_ALLOWED_SOURCES", "MED,PMC").split(",")
    if value.strip()
}

_NCBI_RATE_LOCK = threading.Lock()
_LAST_NCBI_REQUEST_AT = 0.0
_NCBI_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}


class LiteratureProviderUnavailable(RuntimeError):
    """Raised when a provider is unavailable and the caller should fail over."""


@dataclass
class _ProviderState:
    """Run-scoped provider health shared by concurrent candidate searches."""

    ncbi_unavailable: bool = False
    ncbi_error: str | None = None
    europe_pmc_unavailable: bool = False
    europe_pmc_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_ncbi_unavailable(self, error: str) -> None:
        with self.lock:
            self.ncbi_unavailable = True
            self.ncbi_error = error

    def is_ncbi_unavailable(self) -> bool:
        with self.lock:
            return self.ncbi_unavailable

    def mark_europe_pmc_unavailable(self, error: str) -> None:
        with self.lock:
            self.europe_pmc_unavailable = True
            self.europe_pmc_error = error

    def is_europe_pmc_unavailable(self) -> bool:
        with self.lock:
            return self.europe_pmc_unavailable


def _candidate_value(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _clean_term(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_literature_queries(
    candidate: MergedDiseaseCandidateV2,
    hpo_labels: Iterable[str] = (),
    retry: bool = False,
) -> list[str]:
    disease = _clean_term(_candidate_value(candidate, "disease_name"))
    disease_names = [
        _clean_term(part)
        for part in re.split(r"\s*[;|]\s*", disease)
        if _clean_term(part)
    ]
    primary_name = disease_names[0] if disease_names else disease
    omim = canonical_omim_id(_candidate_value(candidate, "OMIM_id"))
    genes = [_clean_term(value) for value in (_candidate_value(candidate, "gene_symbols", []) or []) if _clean_term(value)]
    synonyms = [_clean_term(value) for value in (_candidate_value(candidate, "synonyms", []) or []) if _clean_term(value)]
    phenotypes = [_clean_term(value) for value in (_candidate_value(candidate, "phenotype_labels", []) or []) if _clean_term(value)]
    hpos = [_clean_term(value) for value in hpo_labels if _clean_term(value)]

    queries: list[str] = []
    # Keep identifier and alternate-name searches ahead of broad phenotype
    # searches.  Previously the first two exact display-name queries consumed
    # the default query budget, silently omitting OMIM IDs and synonyms.
    if omim:
        number = re.sub(r"^OMIM:", "", omim, flags=re.IGNORECASE)
        queries.append(f'"{number}"[All Fields]')
    if primary_name:
        queries.append(f'"{primary_name}"')
    if genes:
        query_name = primary_name or disease
        queries.append(f'("{query_name}" OR {" OR ".join(genes[:3])})')
    for synonym in synonyms:
        if synonym.casefold() != primary_name.casefold():
            queries.append(f'"{synonym}"')
    for name in disease_names[1:]:
        if name.casefold() != primary_name.casefold():
            queries.append(f'"{name}"')
    feature_terms = list(dict.fromkeys(phenotypes[:2] + hpos[:2]))
    if genes and feature_terms:
        queries.append(f'({genes[0]}) AND ({" OR ".join(feature_terms)})')
    if primary_name and feature_terms:
        queries.append(f'"{primary_name}" AND ({" OR ".join(feature_terms[:3])})')
    if retry and primary_name:
        queries.extend([
            f'"{primary_name}" AND (genotype OR inheritance OR diagnosis)',
            f'"{primary_name}" AND ("case report" OR cohort OR series)',
        ])
    # A retry gets a small additional budget so the fallback terms are not
    # appended and then immediately truncated by the original limit.
    query_limit = DEFAULT_MAX_QUERIES + (2 if retry else 0)
    return list(dict.fromkeys(query for query in queries if query))[:query_limit]


def _is_ncbi_endpoint(endpoint: str) -> bool:
    return endpoint.startswith(NCBI_BASE_URL)


def _is_literature_endpoint(endpoint: str) -> bool:
    return _is_ncbi_endpoint(endpoint) or endpoint.startswith(EUROPE_PMC_BASE_URL)


def _ncbi_params(params: dict[str, Any]) -> dict[str, Any]:
    """Add the identifiers NCBI requests for automated clients."""
    enriched = dict(params)
    enriched.setdefault("tool", NCBI_TOOL)
    if NCBI_EMAIL:
        enriched.setdefault("email", NCBI_EMAIL)
    if NCBI_API_KEY:
        enriched.setdefault("api_key", NCBI_API_KEY)
    return enriched


def _wait_for_ncbi_slot() -> None:
    """Apply one process-wide E-utilities rate limit across worker threads."""
    global _LAST_NCBI_REQUEST_AT
    interval = 1.0 / NCBI_MAX_REQUESTS_PER_SECOND
    with _NCBI_RATE_LOCK:
        now = time.monotonic()
        wait = interval - (now - _LAST_NCBI_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_NCBI_REQUEST_AT = time.monotonic()


def _request_params(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    if _is_ncbi_endpoint(endpoint):
        _wait_for_ncbi_slot()
        return _ncbi_params(params)
    return dict(params)


def _request_json(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    retries: int = REQUEST_RETRIES,
    on_error=None,
) -> dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            response = session.get(
                endpoint,
                params=_request_params(endpoint, params),
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429 and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt
                except (TypeError, ValueError):
                    delay = 2 ** attempt
                time.sleep(delay)
                continue
            if _is_literature_endpoint(endpoint) and response.status_code in _NCBI_TRANSIENT_STATUS_CODES:
                message = f"{response.status_code} Server Error for url: {getattr(response, 'url', endpoint)}"
                if on_error:
                    on_error(message)
                raise LiteratureProviderUnavailable(message)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt >= retries:
                if on_error:
                    on_error(str(exc))
                if _is_literature_endpoint(endpoint):
                    raise LiteratureProviderUnavailable(str(exc)) from exc
                return {}
            time.sleep(2 ** attempt)
    return {}


def _request_text(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    retries: int = REQUEST_RETRIES,
    on_error=None,
) -> str:
    for attempt in range(retries + 1):
        try:
            response = session.get(
                endpoint,
                params=_request_params(endpoint, params),
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429 and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt
                except (TypeError, ValueError):
                    delay = 2 ** attempt
                time.sleep(delay)
                continue
            if _is_literature_endpoint(endpoint) and response.status_code in _NCBI_TRANSIENT_STATUS_CODES:
                message = f"{response.status_code} Server Error for url: {getattr(response, 'url', endpoint)}"
                if on_error:
                    on_error(message)
                raise LiteratureProviderUnavailable(message)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            if attempt >= retries:
                if on_error:
                    on_error(str(exc))
                if _is_literature_endpoint(endpoint):
                    raise LiteratureProviderUnavailable(str(exc)) from exc
                return ""
            time.sleep(2 ** attempt)
    return ""


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _parse_article(article: ET.Element) -> dict[str, Any]:
    citation = article.find(".//MedlineCitation")
    article_node = citation.find("./Article") if citation is not None else None
    title = _text(article_node.find("./ArticleTitle") if article_node is not None else None)
    abstract_parts = []
    for abstract in article_node.findall("./Abstract/AbstractText") if article_node is not None else []:
        label = abstract.attrib.get("Label")
        text = _text(abstract)
        abstract_parts.append(f"{label}: {text}" if label and text else text)
    abstract = " ".join(part for part in abstract_parts if part).strip()
    journal = _text(article_node.find("./Journal/Title") if article_node is not None else None)
    year = _text(article_node.find("./Journal/JournalIssue/PubDate/Year") if article_node is not None else None)
    if not year:
        year = _text(article_node.find("./Journal/JournalIssue/PubDate/MedlineDate") if article_node is not None else None)[:4]
    authors = []
    for author in article_node.findall("./AuthorList/Author") if article_node is not None else []:
        name = " ".join(filter(None, [_text(author.find("./ForeName")), _text(author.find("./LastName"))]))
        if name:
            authors.append(name)
    article_ids = {}
    for node in article.findall(".//ArticleId"):
        if node.attrib.get("IdType") and node.text:
            article_ids[node.attrib["IdType"].lower()] = node.text.strip()
    pmid = _text(citation.find("./PMID") if citation is not None else None)
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract[:MAX_ABSTRACT_LENGTH],
        "journal": journal,
        "year": year,
        "authors": authors,
        "doi": article_ids.get("doi"),
        "pmc": article_ids.get("pmc"),
    }


def build_europepmc_queries(
    candidate: MergedDiseaseCandidateV2,
    retry: bool = False,
) -> list[str]:
    """Build provider-neutral queries for Europe PMC's REST search API."""
    disease = _clean_term(_candidate_value(candidate, "disease_name"))
    names = [
        _clean_term(part)
        for part in re.split(r"\s*[;|]\s*", disease)
        if _clean_term(part)
    ]
    primary_name = names[0] if names else disease
    genes = [
        _clean_term(value)
        for value in (_candidate_value(candidate, "gene_symbols", []) or [])
        if _clean_term(value)
    ]
    synonyms = [
        _clean_term(value)
        for value in (_candidate_value(candidate, "synonyms", []) or [])
        if _clean_term(value)
    ]
    queries: list[str] = []
    if primary_name:
        queries.append(f'"{primary_name}"')
    if primary_name and genes:
        queries.append(f'"{primary_name}" AND ({" OR ".join(genes[:3])})')
    for synonym in synonyms:
        if synonym.casefold() != primary_name.casefold():
            queries.append(f'"{synonym}"')
    for name in names[1:]:
        if name.casefold() != primary_name.casefold():
            queries.append(f'"{name}"')
    if retry and primary_name:
        queries.append(f'"{primary_name}" AND (genotype OR inheritance OR diagnosis)')
    return list(dict.fromkeys(query for query in queries if query))[:DEFAULT_MAX_QUERIES]


def _strip_html(value: Any) -> str:
    return re.sub(r"<[^>]+>", " ", str(value or "")).strip()


def _parse_europepmc_result(result: dict[str, Any]) -> dict[str, Any]:
    author_list = result.get("authorList") or {}
    authors = []
    for author in author_list.get("author", []) or []:
        name = _clean_term(author.get("fullName") or author.get("authorName"))
        if name:
            authors.append(name)
    if not authors and result.get("authorString"):
        authors = [
            _clean_term(value)
            for value in str(result["authorString"]).split(",")
            if _clean_term(value)
        ]
    pmid = _clean_term(result.get("pmid") or (result.get("source") == "MED" and result.get("id")))
    pmcid = _pmc_identifier(result.get("pmcid"))
    return {
        "pmid": pmid or None,
        "pmc": pmcid,
        "title": _strip_html(result.get("title")),
        "abstract": _strip_html(result.get("abstractText"))[:MAX_ABSTRACT_LENGTH],
        "journal": _clean_term(result.get("journalTitle")),
        "year": _clean_term(result.get("pubYear")),
        "authors": authors,
        "doi": _clean_term(result.get("doi")) or None,
    }


def _parse_pmc_full_text(xml: str) -> dict[str, str]:
    """Extract bounded article text from a PMC full-text XML record.

    PMC EFetch returns JATS-like XML.  We deliberately keep section headings
    and paragraph text as retrieved source content; this is not an LLM
    summary.  Metadata is still taken from the PubMed record so the two
    records remain linked by PMID/PMCID.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}

    body = next(
        (element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "body"),
        None,
    )
    if body is None:
        return {}

    chunks: list[str] = []
    for element in body.iter():
        if element.tag.rsplit("}", 1)[-1] not in {"title", "p"}:
            continue
        text = _text(element)
        if text and (not chunks or chunks[-1] != text):
            chunks.append(text)
    content = "\n\n".join(chunks).strip()
    if not content:
        return {}
    return {"content": content[:MAX_FULL_TEXT_LENGTH]}


def _pmc_identifier(value: Any) -> str | None:
    value = _clean_term(value)
    if not value:
        return None
    return value if value.upper().startswith("PMC") else f"PMC{value}"


def _fetch_pmc_full_text(
    session: requests.Session,
    pmcid: str,
) -> dict[str, str]:
    """Fetch PMC full text through the NCBI E-utilities endpoint.

    PMC documents that E-utilities is an approved automated retrieval
    interface.  This is best-effort: unavailable/restricted records return an
    empty result and the caller retains the PubMed abstract as fallback.
    """
    xml = _request_text(
        session,
        f"{NCBI_BASE_URL}/efetch.fcgi",
        {"db": "pmc", "id": pmcid, "retmode": "xml"},
    )
    return _parse_pmc_full_text(xml) if xml else {}


def _source_keys(source: SourceRecord) -> set[str]:
    keys = set()
    for value in (source.identifier, source.pmid, source.pmcid, source.doi, source.url):
        value = _clean_term(value)
        if value:
            keys.add(value.casefold())
    return keys


def _subtype_numbers(text: str) -> set[str]:
    text = str(text or "").casefold()
    numbers = set(re.findall(r"\b(?:cdls|type|subtype|form)\s*[- ]?(\d+)\b", text))
    # Disease names commonly encode the subtype as "syndrome 1".
    numbers.update(re.findall(r"\b(?:syndrome|disease)\s+(\d+)\b", text))
    return numbers


def _family_terms(text: str) -> list[str]:
    text = re.sub(r"\b(?:cdls|type|subtype|form)\s*[- ]?\d+\b", " ", str(text or "").casefold())
    text = re.sub(r"\b(?:syndrome|disease)\s+\d+\b", " syndrome ", text)
    return [term for term in re.findall(r"[a-z0-9]+", text) if len(term) >= 2]


def matches_candidate_identity(article: dict[str, Any], candidate: MergedDiseaseCandidateV2) -> bool:
    """Reject an article that explicitly belongs to a different subtype.

    Generic papers about a disease family remain eligible.  A paper that says
    CDLS4/type 4, however, must not be attached to the CDLS1 candidate merely
    because both contain the broad phrase "Cornelia de Lange syndrome". Short
    all-caps aliases are matched case-sensitively: BOFS must not match BOFs,
    which is a common abbreviation for bioorganic fertilizers.
    """
    raw_haystack = f"{article.get('title', '')} {article.get('abstract', '')}"
    haystack = raw_haystack.casefold()
    disease_name = _clean_term(_candidate_value(candidate, "disease_name", ""))
    candidate_numbers = _subtype_numbers(disease_name)
    if not candidate_numbers:
        compact_name = re.sub(r"[^a-z0-9]", "", disease_name.casefold())
        compact_markers = set(re.findall(r"cdls(\d+)", compact_name))
        candidate_numbers.update(compact_markers)
    family_terms = _family_terms(disease_name)
    article_numbers = _subtype_numbers(haystack)
    if candidate_numbers and article_numbers and candidate_numbers.isdisjoint(article_numbers):
        if family_terms and all(term in haystack for term in family_terms[:3]):
            return False
    synonyms = [
        _clean_term(value)
        for value in (_candidate_value(candidate, "synonyms", []) or [])
        if _clean_term(value)
    ]
    abbreviations = [
        synonym for synonym in synonyms
        if re.fullmatch(r"[A-Z][A-Z0-9-]{2,7}", synonym)
    ]
    terms = [
        disease_name,
        " ".join(family_terms),
        re.sub(r"^omim:", "", _clean_term(_candidate_value(candidate, "OMIM_id", "")), flags=re.IGNORECASE),
        *(synonym for synonym in synonyms if synonym not in abbreviations),
        *(_candidate_value(candidate, "gene_symbols", []) or []),
    ]
    terms = [_clean_term(term).casefold() for term in terms if _clean_term(term)]
    if any(len(term) >= 4 and term in haystack for term in terms):
        return True

    return any(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            raw_haystack,
        )
        for alias in abbreviations
    )


def _matches_candidate(article: dict[str, Any], candidate: MergedDiseaseCandidateV2) -> bool:
    return matches_candidate_identity(article, candidate)


def _source_id(identifier: str, candidate_id: str) -> str:
    digest = hashlib.sha1(f"{identifier}|{candidate_id}".encode()).hexdigest()[:10]
    return f"SRC-{digest}"


def _source_keys_from_article(article: dict[str, Any]) -> set[str]:
    pmid = _clean_term(article.get("pmid"))
    pmcid = _pmc_identifier(article.get("pmc"))
    doi = _clean_term(article.get("doi"))
    return {
        value.casefold()
        for value in (
            pmid and f"PMID:{pmid}",
            pmcid and f"PMCID:{pmcid}",
            doi,
        )
        if value
    }


def _source_from_article(
    article: dict[str, Any],
    candidate: MergedDiseaseCandidateV2,
    query: str,
    session: requests.Session,
    provider: str,
) -> SourceRecord | None:
    candidate_id = _candidate_value(candidate, "candidate_id")
    pmid = _clean_term(article.get("pmid")) or None
    pmcid = _pmc_identifier(article.get("pmc"))
    doi = _clean_term(article.get("doi")) or None
    if not pmid and not pmcid and not doi:
        return None

    content = _clean_term(article.get("abstract"))
    source_name = "PubMed" if pmid else "Europe PMC"
    content_type = "abstract"
    identifier = f"PMID:{pmid}" if pmid else (f"PMCID:{pmcid}" if pmcid else f"DOI:{doi}")
    source_url = (
        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if pmid
        else f"https://europepmc.org/article/{'PMC' if pmcid else 'MED'}/{pmcid or doi}"
    )
    if pmcid:
        try:
            pmc_content = _fetch_pmc_full_text(session, pmcid)
        except LiteratureProviderUnavailable:
            pmc_content = {}
        if pmc_content.get("content"):
            content = pmc_content["content"]
            source_name = "PMC"
            content_type = "full_text"
            identifier = f"PMCID:{pmcid}"
            source_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

    if not content:
        return None
    return SourceRecord(
        source_id=_source_id(identifier, candidate_id),
        source_type="literature",
        source_name=source_name,
        title=article.get("title") or "Untitled literature record",
        url=source_url,
        identifier=identifier,
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        authors=article.get("authors", []),
        journal=article.get("journal") or None,
        publication_year=article.get("year") or None,
        candidate_ids=[candidate_id],
        disease_name=_candidate_value(candidate, "disease_name"),
        OMIM_id=canonical_omim_id(_candidate_value(candidate, "OMIM_id")),
        gene_symbols=list(_candidate_value(candidate, "gene_symbols", []) or []),
        content_type=content_type,
        raw_content=content,
        search_query=f"{provider}:{query}",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        citation_allowed=True,
    )


def search_candidate_literature_europe_pmc(
    candidate: MergedDiseaseCandidateV2,
    existing_sources: Iterable[SourceRecord] = (),
    retry: bool = False,
    session: requests.Session | None = None,
    provider_state: _ProviderState | None = None,
) -> list[SourceRecord]:
    """Search Europe PMC and normalize results into the same SourceRecord ledger."""
    session = session or requests.Session()
    provider_state = provider_state or _ProviderState()
    if provider_state.is_europe_pmc_unavailable():
        return []
    existing_keys = {
        key
        for source in existing_sources
        if source.source_type == "literature"
        for key in _source_keys(source)
    }
    sources: list[SourceRecord] = []
    diagnostic = {
        "provider": "europe_pmc",
        "candidate_id": _candidate_value(candidate, "candidate_id"),
        "retry": retry,
        "queries": [],
        "accepted_sources": 0,
    }
    for query in build_europepmc_queries(candidate, retry=retry):
        query_diagnostic = {
            "query": query,
            "hits": 0,
            "identity_rejected": 0,
            "source_rejected": 0,
            "identifier_rejected": 0,
            "accepted_sources": 0,
            "errors": [],
        }
        diagnostic["queries"].append(query_diagnostic)
        try:
            payload = _request_json(
                session,
                f"{EUROPE_PMC_BASE_URL}/search",
                {
                    "query": query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": EUROPE_PMC_MAX_RESULTS,
                },
                on_error=query_diagnostic["errors"].append,
            )
        except LiteratureProviderUnavailable as exc:
            provider_state.mark_europe_pmc_unavailable(str(exc))
            diagnostic["provider_error"] = str(exc)
            print(f"[Literature] Europe PMC unavailable: {exc}")
            break
        results = (payload.get("resultList", {}) or {}).get("result", []) or []
        query_diagnostic["hits"] = len(results)
        for result in results:
            provider_source = _clean_term(result.get("source")).upper()
            if provider_source not in EUROPE_PMC_ALLOWED_SOURCES:
                query_diagnostic["source_rejected"] += 1
                continue
            article = _parse_europepmc_result(result)
            if not article.get("pmid") and not article.get("pmc"):
                # DOI-only Europe PMC records can be preprints or records
                # outside PubMed/PMC. They are not stable enough for the
                # clinical literature Reference contract.
                query_diagnostic["identifier_rejected"] += 1
                continue
            if not _matches_candidate(article, candidate):
                query_diagnostic["identity_rejected"] += 1
                continue
            article_keys = _source_keys_from_article(article)
            if not article_keys or article_keys & existing_keys:
                continue
            source = _source_from_article(
                article, candidate, query, session, provider="europe_pmc"
            )
            if source is None:
                continue
            sources.append(source)
            query_diagnostic["accepted_sources"] += 1
            diagnostic["accepted_sources"] += 1
            existing_keys.update(article_keys)
            if len(sources) >= EUROPE_PMC_MAX_RESULTS:
                print(f"[LiteratureSearchDiagnostic] {json.dumps(diagnostic, ensure_ascii=False)}")
                return sources
    print(f"[LiteratureSearchDiagnostic] {json.dumps(diagnostic, ensure_ascii=False)}")
    return sources


def search_candidate_literature(
    candidate: MergedDiseaseCandidateV2,
    hpo_labels: Iterable[str] = (),
    existing_sources: Iterable[SourceRecord] = (),
    retry: bool = False,
    session: requests.Session | None = None,
    provider_state: _ProviderState | None = None,
) -> list[SourceRecord]:
    session = session or requests.Session()
    provider_state = provider_state or _ProviderState()
    if provider_state.is_ncbi_unavailable():
        print(
            "[Literature] PubMed unavailable for this run; "
            f"using Europe PMC for {_candidate_value(candidate, 'candidate_id')}"
        )
        return search_candidate_literature_europe_pmc(
            candidate,
            existing_sources=existing_sources,
            retry=retry,
            session=session,
            provider_state=provider_state,
        )
    candidate_id = _candidate_value(candidate, "candidate_id")
    existing_keys = {
        key
        for source in existing_sources
        if source.source_type == "literature"
        for key in _source_keys(source)
    }
    sources: list[SourceRecord] = []
    diagnostic = {
        "candidate_id": candidate_id,
        "retry": retry,
        "queries": [],
        "accepted_sources": 0,
    }
    for query in build_literature_queries(candidate, hpo_labels, retry=retry):
        query_diagnostic = {
            "query": query,
            "hits": 0,
            "articles_fetched": 0,
            "identity_rejected": 0,
            "duplicate_or_missing_pmid": 0,
            "accepted_sources": 0,
            "errors": [],
        }
        diagnostic["queries"].append(query_diagnostic)
        try:
            search_payload = _request_json(
                session,
                f"{NCBI_BASE_URL}/esearch.fcgi",
                {"db": "pubmed", "term": query, "retmode": "json", "retmax": DEFAULT_MAX_RESULTS},
                on_error=query_diagnostic["errors"].append,
            )
        except LiteratureProviderUnavailable as exc:
            provider_state.mark_ncbi_unavailable(str(exc))
            diagnostic["provider_error"] = str(exc)
            print(f"[Literature] PubMed unavailable; falling back to Europe PMC: {exc}")
            fallback_sources = search_candidate_literature_europe_pmc(
                candidate,
                existing_sources=existing_sources,
                retry=retry,
                session=session,
                provider_state=provider_state,
            )
            diagnostic["fallback_provider"] = "europe_pmc"
            diagnostic["fallback_sources"] = len(fallback_sources)
            print(f"[LiteratureSearchDiagnostic] {json.dumps(diagnostic, ensure_ascii=False)}")
            return fallback_sources
        ids = search_payload.get("esearchresult", {}).get("idlist", [])
        query_diagnostic["hits"] = len(ids)
        if not ids:
            continue
        try:
            xml = _request_text(
                session,
                f"{NCBI_BASE_URL}/efetch.fcgi",
                {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                on_error=query_diagnostic["errors"].append,
            )
        except LiteratureProviderUnavailable as exc:
            provider_state.mark_ncbi_unavailable(str(exc))
            diagnostic["provider_error"] = str(exc)
            print(f"[Literature] PubMed fetch unavailable; falling back to Europe PMC: {exc}")
            fallback_sources = search_candidate_literature_europe_pmc(
                candidate,
                existing_sources=existing_sources,
                retry=retry,
                session=session,
                provider_state=provider_state,
            )
            diagnostic["fallback_provider"] = "europe_pmc"
            diagnostic["fallback_sources"] = len(fallback_sources)
            print(f"[LiteratureSearchDiagnostic] {json.dumps(diagnostic, ensure_ascii=False)}")
            return fallback_sources
        if not xml:
            continue
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            query_diagnostic["errors"].append("PubMed returned invalid XML")
            continue
        article_nodes = root.findall(".//PubmedArticle")
        query_diagnostic["articles_fetched"] = len(article_nodes)
        for article_node in article_nodes:
            article = _parse_article(article_node)
            pmid = article.get("pmid")
            pmcid = _pmc_identifier(article.get("pmc"))
            doi = _clean_term(article.get("doi")) or None
            pubmed_identifier = f"PMID:{pmid}" if pmid else None
            article_keys = {
                value.casefold()
                for value in (pubmed_identifier, pmcid and f"PMCID:{pmcid}", doi)
                if value
            }
            if not pubmed_identifier or article_keys & existing_keys:
                query_diagnostic["duplicate_or_missing_pmid"] += 1
                continue
            if not _matches_candidate(article, candidate):
                query_diagnostic["identity_rejected"] += 1
                continue

            content = article.get("abstract", "")
            source_name = "PubMed"
            content_type = "abstract"
            identifier = pubmed_identifier
            source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            if pmcid:
                try:
                    pmc_content = _fetch_pmc_full_text(session, pmcid)
                except LiteratureProviderUnavailable:
                    pmc_content = {}
                if pmc_content.get("content"):
                    content = pmc_content["content"]
                    source_name = "PMC"
                    content_type = "full_text"
                    identifier = f"PMCID:{pmcid}"
                    source_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

            if not content:
                query_diagnostic["errors"].append(f"No abstract/full text for PMID:{pmid}")
                continue
            source = SourceRecord(
                source_id=_source_id(identifier, candidate_id),
                source_type="literature",
                source_name=source_name,
                title=article.get("title") or "Untitled PubMed article",
                url=source_url,
                identifier=identifier,
                pmid=pmid,
                pmcid=pmcid,
                doi=doi,
                authors=article.get("authors", []),
                journal=article.get("journal") or None,
                publication_year=article.get("year") or None,
                candidate_ids=[candidate_id],
                disease_name=_candidate_value(candidate, "disease_name"),
                OMIM_id=canonical_omim_id(_candidate_value(candidate, "OMIM_id")),
                gene_symbols=list(_candidate_value(candidate, "gene_symbols", []) or []),
                content_type=content_type,
                raw_content=content,
                search_query=query,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                citation_allowed=True,
            )
            sources.append(source)
            query_diagnostic["accepted_sources"] += 1
            diagnostic["accepted_sources"] += 1
            existing_keys.update(article_keys)
            if len(sources) >= DEFAULT_MAX_RESULTS:
                print(f"[LiteratureSearchDiagnostic] {json.dumps(diagnostic, ensure_ascii=False)}")
                return sources
    print(f"[LiteratureSearchDiagnostic] {json.dumps(diagnostic, ensure_ascii=False)}")
    return sources


def search_literature_for_candidates(
    candidates: Iterable[MergedDiseaseCandidateV2],
    hpo_dict: dict[str, str] | None = None,
    existing_sources: Iterable[SourceRecord] = (),
    retry: bool = False,
    retry_candidate_ids: set[str] | None = None,
) -> list[SourceRecord]:
    candidate_list = list(candidates)
    all_sources = list(existing_sources)
    hpo_labels = list((hpo_dict or {}).values())

    if not candidate_list:
        return all_sources

    provider_state = _ProviderState()
    # Each worker gets the same immutable snapshot.  Sharing the growing
    # ``all_sources`` list between workers would make de-duplication racey.
    # Results are merged in candidate order below, preserving the old output
    # order and its first-source-wins behavior.
    base_sources = list(all_sources)
    results_by_index: list[list[SourceRecord]] = [[] for _ in candidate_list]

    def search_one(index: int, candidate: MergedDiseaseCandidateV2) -> tuple[int, list[SourceRecord]]:
        candidate_id = _candidate_value(candidate, "candidate_id")
        if retry and retry_candidate_ids is not None and candidate_id not in retry_candidate_ids:
            print(
                f"[Literature] retry skipped; candidate is not unresolved: {candidate_id}"
            )
            return index, []
        try:
            return index, search_candidate_literature(
                candidate,
                hpo_labels=hpo_labels,
                existing_sources=base_sources,
                retry=retry,
                provider_state=provider_state,
            )
        except Exception as exc:
            print(f"[Literature] search failed for {candidate_id}: {exc}")
            return index, []

    max_workers = min(len(candidate_list), LITERATURE_MAX_WORKERS)
    print(f"[Literature] candidate search max_workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(search_one, index, candidate): index
            for index, candidate in enumerate(candidate_list)
        }
        for future in as_completed(futures):
            index, sources = future.result()
            results_by_index[index] = sources

    # Merge deterministically and retain the previous de-duplication semantics
    # when two candidates return the same PMID/PMCID/DOI.
    seen_literature_keys = {
        key
        for source in all_sources
        if source.source_type == "literature"
        for key in _source_keys(source)
    }
    for candidate_sources in results_by_index:
        for source in candidate_sources:
            source_keys = _source_keys(source)
            if source_keys & seen_literature_keys:
                continue
            all_sources.append(source)
            seen_literature_keys.update(source_keys)
    return all_sources
