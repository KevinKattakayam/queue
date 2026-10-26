"""Identity buffer and spatial recovery tests."""
from queue_abandonment.identity_recovery import IdentityRecovery
from queue_abandonment.state_machine import VisitMachine

def test_nearby_new_track_recovers_old_visit():
    """A nearby ID within timeout links to its prior visit."""
    visit = VisitMachine(3).join(12, 0)
    recovery = IdentityRecovery(2, 40)
    assert recovery.mark_lost(12, (100, 100), 1, visit, True)
    match = recovery.recover(99, (120, 100), 2)
    assert match is not None
    assert visit.track_id == 99
    assert visit.previous_track_ids == [12]

def test_distant_track_does_not_reassociate():
    """A detection outside the configured radius remains a different person."""
    visit = VisitMachine(3).join(1, 0)
    recovery = IdentityRecovery(2, 20)
    recovery.mark_lost(1, (0, 0), 0, visit, True)
    assert recovery.recover(2, (100, 100), 1) is None

def test_expired_buffer_is_removed():
    """Lost IDs expire after the configured timeout."""
    visit = VisitMachine(3).join(1, 0)
    recovery = IdentityRecovery(1, 20)
    recovery.mark_lost(1, (0, 0), 0, visit, True)
    assert len(recovery.expire(2)) == 1

def test_queue_interior_disappearance_is_not_edge_departure():
    """An interior disappearance in the queue is treated as occlusion."""
    assert not IdentityRecovery.likely_departure((100, 100), (500, 500), True, 30)

def test_edge_disappearance_is_departure():
    """An edge disappearance is treated as likely leaving."""
    assert IdentityRecovery.likely_departure((2, 200), (500, 500), True, 30)
