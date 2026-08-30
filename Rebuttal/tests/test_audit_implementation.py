from pathlib import Path

from Rebuttal.tests.audit_implementation import audit


def test_current_rebuttal_implementation_passes_aggregate_audit() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = audit(repo_root)
    assert result["passed"], result["errors"]
    assert "trace content" in result["privacy_boundary"]
    assert result["sections"]["coverage"]["trajectory_counts"] == {
        "Browser": 16,
        "File": 16,
    }
    assert result["sections"]["coverage"]["primary"]["trajectory_counts"] == {
        "Browser": 72,
        "File": 60,
    }
    assert (
        result["sections"]["coverage"]["primary"]["empirical_result"] is False
    )
    assert result["sections"]["coverage"][
        "supplemental_16_target"
    ]["trajectory_counts"] == {"Browser": 16, "File": 16}
    assert result["sections"]["materialization"]["status"] == "materialized"
