from trendlens.schemas import ClassifiedRecord, Experiment, Opportunity


def test_classified_record_dedupes_themes():
    rec = ClassifiedRecord(id="p1", themes=["taste", "taste", "health"], sentiment="pos")
    assert rec.themes == ["taste", "health"]


def test_opportunity_score_clamped():
    assert Opportunity(theme="t", score=1.5).score == 1.0
    assert Opportunity(theme="t", score=-0.3).score == 0.0


def test_experiment_has_no_winner_field():
    exp = Experiment(
        hypothesis="h", a_copy="a", b_copy="b", audience="aud", metric="ctr"
    )
    assert not hasattr(exp, "predicted_winner")
