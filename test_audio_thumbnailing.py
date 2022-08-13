import sys
import json

sys.path.insert(0, './src')
from Audio_thumbnailing import AudioThumbnailer


def test_one():
    t = AudioThumbnailer('Monk2_master.mp3')
    t.run()

    print(json.dumps(t.thumbnail, indent=2))

    assert t.thumbnail == {
        "filename": "Monk2_master.mp3",
        "thumbnail": {
            "boundaries_in_seconds": "[[20.0, 50.0], [145.5, 172.5]]",
            "fitness": "0.135",
            "nominal_duration_in_seconds": 30,
            "search_step_in_seconds": 5,
            "total_coverage_in_seconds": 58.0,
            "normalized_total_coverage": "0.152",
            "score": "74.747",
            "normalized_score": "0.122"
        },
        "context": {
            "audio_duration_seconds": "181.16",
            "feature_rate_hz": 2.0,
            "ssm_dimensions": {
                "x": 363,
                "y": 363
            }
        }
    }
