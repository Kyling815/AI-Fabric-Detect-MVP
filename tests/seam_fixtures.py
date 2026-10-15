from datetime import datetime, timezone

from backend.app.core.physical_coordinates import PixelCandidate
from backend.app.core.seam import SeamObservation, SeamPolicy
from tests.physical_coordinate_fixtures import calibration, frame


def seam_policy(**updates):
    fields = dict(adjacent_camera_pairs=(("lane_left", "lane_right"),))
    fields.update(updates)
    return SeamPolicy(**fields)


def observation(camera="lane_left", *, x=None, y=8, w=4, h=4, label="defect", candidate_id="candidate", score=0.94,
                frame_updates=None, **updates):
    fields = dict(capture_group_id="fixture_capture_1", timestamp=datetime(2026, 10, 15, 2, tzinfo=timezone.utc),
        source_kind="supervised", model_version="hypothetical_model_v1", model_sha256="a" * 64,
        frame=frame(camera=camera, **(frame_updates or {})),
        candidate=PixelCandidate(candidate_id=candidate_id, x=(42 if camera == "lane_left" else 2) if x is None else x,
            y=y, w=w, h=h, score=score, label=label))
    fields.update(updates)
    return SeamObservation(**fields)


def duplicate_pair():
    return [observation(), observation(camera="lane_right")]
