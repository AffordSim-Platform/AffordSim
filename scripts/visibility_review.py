"""Trial-specific user reviews of automatic camera-framing results."""


def user_confirmed_visibility(trial, task_id, attempt):
    """A review applies only to the exact task, attempt and trial reviewed."""
    review = trial.get('manual_visibility_review')
    return (isinstance(review, dict) and review.get('source') == 'user'
            and review.get('all_objects_in_frame') is True
            and isinstance(task_id, str) and task_id == review.get('task_id')
            and isinstance(attempt, str) and attempt == review.get('attempt')
            and type(trial.get('trial')) is int and trial['trial'] == review.get('trial'))


def user_accepted_preview(trial, task_id, attempt):
    """Accept the reviewed GIF without rewriting the automatic framing result."""
    review = trial.get('manual_preview_review')
    return (isinstance(review, dict) and review.get('source') == 'user'
            and review.get('decision') == 'accepted'
            and isinstance(task_id, str) and task_id == review.get('task_id')
            and isinstance(attempt, str) and attempt == review.get('attempt')
            and type(trial.get('trial')) is int and trial['trial'] == review.get('trial'))


def user_allowed_partial_visibility(trial, task_id, attempt):
    """A task-specific permission does not assert that the object is in frame."""
    review = trial.get('preview_visibility_exception')
    return (isinstance(review, dict) and review.get('source') == 'user'
            and review.get('decision') == 'allow_partial'
            and isinstance(task_id, str) and task_id == review.get('task_id')
            and isinstance(attempt, str) and attempt == review.get('attempt')
            and type(trial.get('trial')) is int and trial['trial'] == review.get('trial'))


def user_requested_fixed_pose_recording(trial, task_id, attempt):
    """Identify a fixed-pose recording without claiming its view is complete."""
    record = trial.get('fixed_pose_recording')
    return (isinstance(record, dict) and record.get('source') == 'user'
            and record.get('task_id') == task_id and record.get('attempt') == attempt
            and type(trial.get('trial')) is int and record.get('trial') == trial['trial'])
