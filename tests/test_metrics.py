from kvagent.metrics import CacheSnapshot, parse_prometheus


def test_parse_prometheus_sums_labels() -> None:
    text = '''
# HELP vllm:prefix_cache_hits Prefix hits.
vllm:prefix_cache_hits_total{model_name="a",engine="0"} 120
vllm:prefix_cache_hits_total{model_name="a",engine="1"} 30
vllm:prefix_cache_queries_total{model_name="a"} 200
'''
    values = parse_prometheus(text)
    assert values["vllm:prefix_cache_hits_total"] == 150
    assert values["vllm:prefix_cache_queries_total"] == 200


def test_snapshot_delta_and_rate() -> None:
    before = CacheSnapshot(100, 25, 500, 25)
    after = CacheSnapshot(300, 175, 800, 175)
    delta = after.delta(before)
    assert delta.prefix_queries == 200
    assert delta.prefix_hits == 150
    assert delta.hit_rate == 0.75
