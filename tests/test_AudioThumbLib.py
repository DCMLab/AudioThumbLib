import sys
import json

sys.path.insert(0, '../src')
from AudioThumbLib import AudioThumbnailer


def test_one():
    t = AudioThumbnailer('Monk.mp3')
    t.run()

    print(json.dumps(t.thumbnail, indent=2))

    assert t.thumbnail == {
            "filename": "Monk.mp3",
            "thumbnail": {
                "boundaries_in_seconds": "[[15.0, 29.0], [33.0, 49.0], [49.5, 61.0], [72.5, 89.5], [155.0, 172.0]]",
                "fitness": "0.354",
                "nominal_duration_in_seconds": 17,
                "search_step_in_seconds": 5,
                "thumbnail_duration_step_in_seconds": 1,
                "coverage_in_seconds": 78.0,
                "normalized_coverage": "0.333",
                "score": "94.002",
                "normalized_score": "0.378"
            },
            "context": {
                "audio_duration_in_seconds": "181.16",
                "feature_rate": 2.0,
                "ssm_dimensions": {
                    "x": 363,
                    "y": 363
                }
            }
        }