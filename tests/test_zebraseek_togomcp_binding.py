from agent.tools.zebraseek_flow import (
    bind_togomcp_actions,
    build_patient_input,
    _fixed_togomcp_args,
)
from agent.tools.togomcp_search import discover_diseases_with_togomcp


def _spec(name):
    return {"name": name, "description": "", "input_schema": {"properties": {}}}


def test_fixed_bindings_do_not_select_pubcasefinder_or_usage_guide():
    specs = [
        _spec("TogoMCP_Usage_Guide"),
        _spec("pubcasefinder_rank_by_phenotypes"),
        _spec("ncbi_esearch"),
        _spec("run_sparql"),
        _spec("get_MIE_file"),
    ]

    bindings = bind_togomcp_actions(specs)

    assert bindings["check_present_hpo"]["tool_name"] == "ncbi_esearch"
    assert bindings["check_absent_hpo"]["tool_name"] == "ncbi_esearch"
    assert bindings["search_case_reports"]["tool_name"] == "ncbi_esearch"
    assert bindings["search_pubmed"]["tool_name"] == "ncbi_esearch"
    assert bindings["expand_candidates"]["tool_name"] is None
    assert bindings["expand_candidates"]["status"] == "derived"
    assert all(
        binding["tool_name"] != "pubcasefinder_rank_by_phenotypes"
        for binding in bindings.values()
    )
    assert all(
        binding["tool_name"] != "TogoMCP_Usage_Guide"
        for binding in bindings.values()
    )


def test_ncbi_fixed_arguments_use_disease_or_literature_databases():
    patient = build_patient_input(
        ["HP:0001263"], absent_hpo_ids=["HP:0001250"], patient_id="p1"
    )
    candidate = {
        "candidate_id": "OMIM:123456",
        "disease_name": "Example syndrome",
        "identifiers": {"omim_id": "OMIM:123456"},
    }

    identity = _fixed_togomcp_args("resolve_identity", candidate, patient, "ncbi_esearch")
    literature = _fixed_togomcp_args("search_pubmed", candidate, patient, "ncbi_esearch")
    case_reports = _fixed_togomcp_args(
        "search_case_reports", candidate, patient, "ncbi_esearch"
    )

    assert identity["database"] == "medgen"
    assert literature["database"] == "pubmed"
    assert case_reports["database"] == "pubmed"
    assert "hpo_ids" not in case_reports
    assert "target" not in case_reports


def test_legacy_togomcp_discovery_does_not_rerun_pcf():
    result = discover_diseases_with_togomcp(
        {"use_togomcp": True, "hpoList": ["HP:0001263"]}
    )
    assert result["togomcp_search_status"] == "legacy_pcf_route_disabled"
    assert result["discovered_disease_candidates"] == []
