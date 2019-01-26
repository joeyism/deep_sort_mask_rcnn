from deep_sort.track import TrackState


def remove_deleted_tracks(tracker):
    for track in tracker.tracks:
        if track.state = TrackState.Deleted:
            del track
