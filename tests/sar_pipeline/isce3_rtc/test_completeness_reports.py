"""Offline tests for the completeness report logic. The CDSE and STAC APIs are mocked,
so these do not require credentials or network access."""

from datetime import datetime
import json
from pathlib import Path

import pytest
import requests
from shapely.geometry import box, MultiPolygon, Point, Polygon

from sar_pipeline.pipelines.isce3_rtc.monitoring import (
    generate_s1_iw_completeness_report as report_module,
)
from sar_pipeline.pipelines.isce3_rtc.utils import burst_utils
from sar_pipeline.pipelines.isce3_rtc.utils.burst_utils import (
    CDSE_ODATA_MAX_RESULTS,
    CDSEQueryFailedError,
    CDSEQueryTruncatedError,
    _get_cdse_request_with_retries,
    _make_chunk_cdse_request,
    query_cdse_for_bursts_in_period,
)
from sar_pipeline.utils.spatial import wkt_to_geojson

TEST_GEOMETRY = MultiPolygon([box(140, -40, 150, -30)])
TEST_ROI_WKT = "POLYGON ((140 -40, 150 -40, 150 -30, 140 -30, 140 -40))"

SCENE_A = "S1A_IW_SLC__1SDV_20241214T050000_20241214T050030_056000_06D000_AAAA"
SCENE_B = "S1A_IW_SLC__1SDV_20241214T050030_20241214T050100_056000_06D000_BBBB"


class _MockResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"value": []}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} error", response=self
            )

    def json(self):
        return self._payload


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(burst_utils.time, "sleep", lambda s: None)


def _mock_requests_get(monkeypatch, responses):
    """Patch requests.get to return/raise the given responses in order."""
    calls = []

    def _get(url, timeout=None):
        calls.append(url)
        response = responses[min(len(calls), len(responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(burst_utils.requests, "get", _get)
    return calls


def _make_cdse_burst(burst_id, az_time, scene):
    return {
        "RelativeOrbitNumber": 70,
        "BurstId": burst_id,
        "SwathIdentifier": "IW1",
        "AzimuthTime": az_time,
        "Name": f"burst_{burst_id}",
        "ParentProductName": f"{scene}.SAFE",
        "PlatformSerialIdentifier": "A",
    }


def test_cdse_request_retries_on_rate_limit(monkeypatch, no_sleep):
    calls = _mock_requests_get(
        monkeypatch,
        [
            _MockResponse(429),
            requests.exceptions.ConnectionError("reset"),
            _MockResponse(200, {"value": [1]}),
        ],
    )
    assert _get_cdse_request_with_retries("url") == {"value": [1]}
    assert len(calls) == 3


def test_cdse_request_raises_after_max_retries(monkeypatch, no_sleep):
    calls = _mock_requests_get(monkeypatch, [_MockResponse(503)])
    with pytest.raises(requests.exceptions.HTTPError):
        _get_cdse_request_with_retries("url", max_retries=2)
    assert len(calls) == 3


def test_cdse_request_does_not_retry_client_error(monkeypatch, no_sleep):
    calls = _mock_requests_get(monkeypatch, [_MockResponse(400)])
    with pytest.raises(requests.exceptions.HTTPError):
        _get_cdse_request_with_retries("url")
    assert len(calls) == 1


def test_chunk_request_splits_when_truncated(monkeypatch):
    # full first response, then each half returns fewer than the limit
    calls = _mock_requests_get(
        monkeypatch,
        [
            _MockResponse(200, {"value": [{}] * CDSE_ODATA_MAX_RESULTS}),
            _MockResponse(200, {"value": [{}] * 600}),
            _MockResponse(200, {"value": [{}] * 500}),
        ],
    )
    results = _make_chunk_cdse_request(
        "url",
        datetime(2024, 12, 14, 5),
        datetime(2024, 12, 14, 6),
        "IW_SLC__1S",
        MultiPolygon([box(140, -40, 150, -30)]),
    )
    assert len(calls) == 3
    assert len(results) == 1100
    assert "ContentDate/Start le 2024-12-14T05:30:00" in calls[1]
    assert "ContentDate/Start ge 2024-12-14T05:30:00" in calls[2]


def test_chunk_request_raises_when_truncated_after_splitting(monkeypatch):
    _mock_requests_get(
        monkeypatch,
        [_MockResponse(200, {"value": [{}] * CDSE_ODATA_MAX_RESULTS})],
    )
    with pytest.raises(CDSEQueryTruncatedError):
        _make_chunk_cdse_request(
            "url",
            datetime(2024, 12, 14, 5),
            datetime(2024, 12, 14, 5, 5),
            "IW_SLC__1S",
            TEST_GEOMETRY,
        )


def test_chunk_request_without_geometry(monkeypatch):
    calls = _mock_requests_get(monkeypatch, [_MockResponse(200, {"value": [{}]})])
    results = _make_chunk_cdse_request(
        "url",
        datetime(2024, 12, 14, 5),
        datetime(2024, 12, 14, 5, 5),
        "IW_SLC__1S",
        None,
    )
    assert len(results) == 1
    assert "Intersects" not in calls[0]


def test_query_bursts_raises_if_any_chunk_fails(monkeypatch):
    def _chunk_request(base_url, chunk_start, chunk_end, product_type, geometry):
        if chunk_start.minute >= 5:
            raise requests.exceptions.HTTPError("503 error")
        return [_make_cdse_burst(1, "2024-12-14T05:00:01.000000Z", SCENE_A)]

    monkeypatch.setattr(burst_utils, "_make_chunk_cdse_request", _chunk_request)
    with pytest.raises(CDSEQueryFailedError, match="1 of 2 CDSE chunk queries failed"):
        query_cdse_for_bursts_in_period(
            start_dt=datetime(2024, 12, 14, 5),
            end_dt=datetime(2024, 12, 14, 5, 10),
            chunk_query_minutes=5,
            geometry=TEST_GEOMETRY,
            query_overlap_seconds=0,
        )


class _MockStacItems:
    def __init__(self, items):
        self.items = items

    def items_as_dicts(self):
        return iter(self.items)


def _make_stac_item(burst_id, az_time):
    return {
        "id": f"{burst_id}_{az_time}",
        "properties": {
            "sarard:burst_id": burst_id,
            "datetime": az_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        },
    }


def test_scene_report_flags_partially_processed_scenes(monkeypatch, tmp_path: Path):
    """A scene is only complete if all expected bursts are indexed. Previously
    a single indexed burst marked the whole scene as processed."""
    az_times = [datetime(2024, 12, 14, 5, 0, s) for s in range(0, 12, 3)]
    expected = {
        ("t070_000001_iw1", az_times[0]): SCENE_A,
        ("t070_000002_iw1", az_times[1]): SCENE_A,
        ("t070_000003_iw1", az_times[2]): SCENE_B,
        ("t070_000004_iw1", az_times[3]): SCENE_B,
    }
    expected_burst_product_dict = {
        key: {"burst_id": key[0], "azimuth_time": key[1], "scene_id": scene}
        for key, scene in expected.items()
    }
    # all of scene A is indexed, but only one burst of scene B
    indexed = [
        _make_stac_item("t070_000001_iw1", az_times[0]),
        _make_stac_item("t070_000002_iw1", az_times[1]),
        _make_stac_item("t070_000003_iw1", az_times[2]),
    ]

    monkeypatch.setattr(
        report_module,
        "query_cdse_for_bursts_in_period",
        lambda **kwargs: expected_burst_product_dict,
    )
    stac_query_kwargs = {}

    def _query_stac(**kwargs):
        stac_query_kwargs.update(kwargs)
        return len(indexed), _MockStacItems(indexed)

    monkeypatch.setattr(report_module, "query_stac_for_metadata_in_period", _query_stac)
    monkeypatch.chdir(tmp_path)

    report_module.make_scene_completeness_report(
        s3_bucket="test-bucket",
        s3_project_folder="test/project",
        collection_number="1",
        start_dt=datetime(2024, 12, 14, 5),
        end_dt=datetime(2024, 12, 14, 6),
        roi_geojson=wkt_to_geojson(TEST_ROI_WKT),
        stac_catalog="https://example.com/stac",
        dry_run=True,
    )

    # products on the edge of the ROI must be returned by the STAC search
    assert stac_query_kwargs["geometry"].contains(Point(150.2, -35))

    report_paths = list((tmp_path / "TMP/monitoring").glob("*.json"))
    assert len(report_paths) == 1
    report = json.loads(report_paths[0].read_text())

    assert report["results"]["scenes_to_reprocess"] == [SCENE_B]
    assert report["results"]["burst_products_missing_by_scene"] == {
        SCENE_B: ["t070_000004_iw1"]
    }
    assert report["summary"] == {
        "n_expected_processed_scenes": 2,
        "n_scenes_to_reprocess": 1,
        "n_expected_burst_products": 4,
        "n_burst_products_missing_from_odc": 1,
    }


def test_expand_geometry_for_product_search_clips_to_valid_coordinates():
    # e.g. the Antarctic ROI, bounded by the antimeridian and the south pole
    roi = Polygon([(-179.99, -50), (179.99, -50), (179.99, -90), (-179.99, -90)])
    expanded = report_module._expand_geometry_for_product_search(roi)
    assert expanded.contains(Point(0, -49.8))
    assert expanded.bounds[0] >= -180 and expanded.bounds[2] <= 180
    assert expanded.bounds[1] >= -90
