from ac_rl.shared_memory import SPageFilePhysics, SPageFileGraphics, SPageFileStatic

def test_physics_struct_has_required_fields():
    assert hasattr(SPageFilePhysics, "packetId")
    assert hasattr(SPageFilePhysics, "speedKmh")
    assert hasattr(SPageFilePhysics, "velocity")
    assert hasattr(SPageFilePhysics, "heading")
    assert hasattr(SPageFilePhysics, "numberOfTyresOut")

def test_graphics_struct_has_required_fields():
    assert hasattr(SPageFileGraphics, "normalizedCarPosition")
    assert hasattr(SPageFileGraphics, "completedLaps")
    assert hasattr(SPageFileGraphics, "carCoordinates")

def test_static_struct_has_track_width():
    assert hasattr(SPageFileStatic, "trackName")
