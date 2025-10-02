# modules/filter.py

def select_target(tracked_objects, target_id=None, target_class=None):
    """
    Filter tracked objects to select a single target
    :param tracked_objects: list of tuples from tracker:
                            (track_id, bbox, class_id, class_name, conf)
    :param target_id: if provided, only keep this track_id
    :param target_class: if provided, only consider objects of this class
    :return: list containing a single target object or empty list
    """
    # 1️⃣ Filter by class if specified
    candidates = tracked_objects
    if target_class is not None:
        candidates = [t for t in candidates if t[3] == target_class]

    # 2️⃣ If target_id is specified, keep only that object
    if target_id is not None:
        candidates = [t for t in candidates if t[0] == target_id]

    # 3️⃣ If no target_id, pick first candidate (or closest to frame center if desired)
    if target_id is None and candidates:
        target = candidates[0]  # simple default: pick first
        return [target]

    return candidates
