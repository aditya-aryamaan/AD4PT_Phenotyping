import pytest

from ad4pt.counting import StrawberryCounter
from ad4pt.vision.detector import Detection


def fruit(identity=1, label="strawberry-ripe", confidence=0.9, x=0):
    return Detection(
        0 if label == "strawberry-ripe" else 1,
        label,
        confidence,
        (x, 0, x + 10, 10),
        track_id=identity,
    )


def test_count_requires_three_observations():
    counter = StrawberryCounter()
    for _ in range(2):
        counter.update((fruit(),))
        assert counter.counts == {"ripe": 0, "unripe": 0}
    counter.update((fruit(),))
    assert counter.counts == {"ripe": 1, "unripe": 0}
    assert counter.frames_processed == 3


def test_single_frame_false_positive_never_counts():
    counter = StrawberryCounter()
    counter.update((fruit(),))
    for _ in range(40):
        counter.update(())
    assert counter.counts == {"ripe": 0, "unripe": 0}


def test_two_fruits_and_motion_do_not_recount():
    counter = StrawberryCounter()
    for x in range(20):
        counter.update((fruit(x=x), fruit(2, "strawberry-unripe", x=x + 30)))
    assert counter.counts == {"ripe": 1, "unripe": 1}


def test_returning_same_id_after_gap_is_not_counted_again():
    counter = StrawberryCounter()
    for _ in range(3):
        counter.update((fruit(),))
    for _ in range(20):
        counter.update(())
    for _ in range(3):
        counter.update((fruit(),))
    assert counter.counts["ripe"] == 1
    for _ in range(3):
        counter.update((fruit(2),))
    assert counter.counts["ripe"] == 2


def test_classification_votes_move_count_without_duplicating():
    counter = StrawberryCounter()
    for _ in range(3):
        counter.update((fruit(label="strawberry-unripe", confidence=0.3),))
    assert counter.counts == {"ripe": 0, "unripe": 1}
    for _ in range(2):
        counter.update((fruit(confidence=0.9),))
    assert counter.counts == {"ripe": 1, "unripe": 0}


def test_untracked_detections_are_not_counted():
    counter = StrawberryCounter()
    for _ in range(5):
        counter.update((fruit(None),))
    assert counter.counts["ripe"] == 0


def test_duplicate_id_cannot_confirm_in_one_frame():
    counter = StrawberryCounter()
    with pytest.raises(ValueError, match="duplicate"):
        counter.update((fruit(), fruit()))
    assert counter.frames_processed == 0


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_confirmation(value):
    with pytest.raises(ValueError):
        StrawberryCounter(value)


def test_weak_observations_cannot_confirm_or_change_ripeness():
    counter = StrawberryCounter()
    for _ in range(10):
        counter.update((fruit(confidence=0.19),))
    assert counter.counts == {"ripe": 0, "unripe": 0}
    for _ in range(3):
        counter.update((fruit(label="strawberry-unripe", confidence=0.32),))
    assert counter.counts == {"ripe": 0, "unripe": 1}
    for _ in range(30):
        counter.update((fruit(confidence=0.19),))
    assert counter.counts == {"ripe": 0, "unripe": 1}


def test_static_scene_ignores_ids_and_outvotes_transient_false_counts():
    from ad4pt.counting import StaticSceneCounter

    counter = StaticSceneCounter()
    for index in range(20):
        scene = tuple(fruit(index * 4 + k + 1, x=k * 15) for k in range(4))
        if index in (5, 8, 15):
            scene += (fruit(999, "strawberry-unripe"),)
        counter.update(scene)
    assert counter.counts == {"ripe": 4, "unripe": 0}
    assert counter.support_frames == 17
    assert counter.frames_processed == 20


def test_static_scene_counts_raw_detections_but_not_weak_ones():
    from ad4pt.counting import StaticSceneCounter

    counter = StaticSceneCounter()
    assert counter.counts == {"ripe": 0, "unripe": 0}
    for _ in range(2):
        counter.update((fruit(None), fruit(None, "strawberry-unripe", 0.19)))
        assert counter.counts == {"ripe": 0, "unripe": 0}
    counter.update((fruit(None),))
    assert counter.counts == {"ripe": 1, "unripe": 0}


def test_static_scene_is_modal_pair_not_largest_count_and_keeps_persistent_errors():
    from ad4pt.counting import StaticSceneCounter

    counter = StaticSceneCounter()
    for _ in range(10):
        counter.update((fruit(None), fruit(None, "strawberry-unripe")))
    counter.update(tuple(fruit(None, x=i * 15) for i in range(20)))
    assert counter.counts == {"ripe": 1, "unripe": 1}
    for _ in range(12):
        counter.update(())
    assert counter.counts == {"ripe": 0, "unripe": 0}


def test_scene_confirmation_rejects_flicker_even_when_it_occurs_in_most_frames():
    from ad4pt.counting import SceneDetectionFilter, StaticSceneCounter

    gate = SceneDetectionFilter(5)
    counter = StaticSceneCounter()
    for index in range(60):
        detections = tuple(fruit(None, x=i * 20) for i in range(4))
        # Leaf flashes in four out of five frames: raw modal counts would fail.
        if index % 5 != 4:
            detections += (fruit(None, "strawberry-unripe", x=100),)
        stable = gate.update(detections)
        assert not any(d.label.endswith("unripe") for d in stable)
        counter.update(stable)
    assert counter.counts == {"ripe": 4, "unripe": 0}


def test_scene_confirmation_preserves_sustained_low_confidence_green_fruit():
    from ad4pt.counting import SceneDetectionFilter

    gate = SceneDetectionFilter(5)
    for index in range(5):
        # Slight movement is allowed; confidence is not raised above 0.25.
        detection = fruit(None, "strawberry-unripe", confidence=0.32, x=index)
        stable = gate.update((detection,))
        assert stable == ((detection,) if index == 4 else ())
    assert gate.update(()) == ()
    assert gate.update((detection,)) == ()  # Missing one frame reset the streak.


def test_scene_confirmation_cannot_combine_different_leaves_or_classes():
    from ad4pt.counting import SceneDetectionFilter

    gate = SceneDetectionFilter(3)
    for index in range(10):
        assert gate.update((fruit(None, x=100 * (index % 2)),)) == ()
    for index in range(10):
        label = "strawberry-ripe" if index % 2 else "strawberry-unripe"
        assert gate.update((fruit(None, label),)) == ()


def test_scene_confirmation_matching_is_one_to_one():
    from ad4pt.counting import SceneDetectionFilter

    gate = SceneDetectionFilter(2)
    assert gate.update((fruit(None),)) == ()
    confirmed = gate.update((fruit(None), fruit(None, x=1)))
    assert len(confirmed) == 1  # Two new boxes cannot both inherit one streak.


@pytest.mark.parametrize("value", [0, -1, 61, True, 1.5])
def test_scene_confirmation_validation(value):
    from ad4pt.counting import SceneDetectionFilter

    with pytest.raises(ValueError):
        SceneDetectionFilter(value)
