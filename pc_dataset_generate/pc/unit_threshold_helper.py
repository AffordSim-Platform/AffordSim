import select
import sys
import termios
import time
import tty

try:
    import open3d as o3d
except ImportError:
    o3d = None


DEFAULT_UNIT_THRESHOLD = 100.0
_VISUALIZATION_KEY = "_open3d_visualization"

# GLFW  ( Open3D VisualizerWithKeyCallback.register_key_callback )
_KEY_ENTER = 257
_KEY_NUMPAD_ENTER = 335
_KEY_ESCAPE = 256
_KEY_ARROW_UP = 265
_KEY_ARROW_DOWN = 264


# ============================================================================
# 
# ============================================================================
def scale_points_with_threshold(raw_points, threshold):
    coord_range = raw_points.ptp(axis=0).max()
    if coord_range > threshold:
        return raw_points / 1000.0, coord_range, True
    return raw_points.copy(), coord_range, False


def print_unit_status(item_name, coord_range, threshold, converted):
    action = "" if converted else ""
    print(
        f"[INFO] {item_name}: coord_range={coord_range:.6f}, "
        f"={threshold:g}, {action}"
    )


# ============================================================================
#   to   
# ============================================================================
def _classify_key(raw):
    """( escape ),  None"""
    if raw is None or raw == "":
        return None
    if raw in ("\r", "\n"):
        return "accept"
    if raw == "\x1b":
        return "quit"
    if raw == "\x1b[A":
        return "up"
    if raw == "\x1b[B":
        return "down"
    low = raw.lower()
    if low == "y":
        return "accept"
    if low == "u":
        return "up"
    if low == "d":
        return "down"
    if low in ("q", "x"):
        return "quit"
    return None


def _read_key_event_nonblock():
    """ cbreak ,  raw  None

     ESC  ( \x1b[A / \x1b[B)  ESC
    """
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    # ,  20ms 
    if not select.select([sys.stdin], [], [], 0.02)[0]:
        return ch  #  ESC
    ch2 = sys.stdin.read(1)
    if ch2 != "[":
        return ch + ch2
    if not select.select([sys.stdin], [], [], 0.02)[0]:
        return ch + ch2
    ch3 = sys.stdin.read(1)
    return ch + ch2 + ch3


def _drain_stdin():
    """ stdin  ()"""
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
    except Exception:
        pass


# ============================================================================
# 
# ============================================================================
def _instructions(threshold):
    return (
        "[INPUT] : "
        "Enter/Y= | "
        f"Up/U=x10  to  {threshold * 10:g} | "
        f"Down/D=/10  to  {threshold / 10:g} | "
        "Q/Esc/X="
    )


def prompt_threshold_decision(threshold):
    """ (), """
    print(_instructions(threshold))

    if not sys.stdin.isatty():
        #  tty  input(): 
        while True:
            raw = input("> ").strip()
            decision = _classify_key(raw[:1] if raw else "\n")
            if decision is not None:
                return decision
            print("  [WARN] ,  (Y / U / D / Q , )")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        _drain_stdin()
        while True:
            key = _read_key_event_nonblock()
            if key is None:
                time.sleep(0.02)
                continue
            decision = _classify_key(key)
            if decision is not None:
                return decision
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ============================================================================
#  payload  / 
# ============================================================================
def attach_visualization(payload, geometries, window_name, width=1280, height=720):
    if not isinstance(payload, dict):
        raise TypeError("payload  dict")

    payload = dict(payload)
    payload[_VISUALIZATION_KEY] = {
        "geometries": list(geometries),
        "window_name": window_name,
        "width": width,
        "height": height,
    }
    return payload


def _split_visualization_payload(payload):
    if not isinstance(payload, dict):
        return payload, None

    payload = dict(payload)
    visualization = payload.pop(_VISUALIZATION_KEY, None)
    return payload, visualization


# ============================================================================
#  + 
#
# :  " Open3D "  "" 
#  poll_events() ,  stdin
#  Linux  Open3D  key callback , 
# 
# ============================================================================
def prompt_visual_decision(threshold, visualization):
    if o3d is None:
        raise ImportError(" open3d, ")

    state = {"decision": None}

    def _make_cb(decision_value):
        def _cb(_vis):
            if state["decision"] is None:
                state["decision"] = decision_value
            return False
        return _cb

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{visualization['window_name']} | : Y= UpDown= Q=",
        width=visualization["width"],
        height=visualization["height"],
    )
    for geometry in visualization["geometries"]:
        vis.add_geometry(geometry)

    accept_cb = _make_cb("accept")
    up_cb = _make_cb("up")
    down_cb = _make_cb("down")
    quit_cb = _make_cb("quit")

    for key in (_KEY_ENTER, _KEY_NUMPAD_ENTER, ord("Y")):
        vis.register_key_callback(key, accept_cb)
    for key in (_KEY_ARROW_UP, ord("U")):
        vis.register_key_callback(key, up_cb)
    for key in (_KEY_ARROW_DOWN, ord("D")):
        vis.register_key_callback(key, down_cb)
    for key in (_KEY_ESCAPE, ord("Q"), ord("X")):
        vis.register_key_callback(key, quit_cb)

    print(_instructions(threshold))
    print("[INFO]  Open3D ; , ")

    fd = None
    old_settings = None
    stdin_is_tty = sys.stdin.isatty()
    if stdin_is_tty:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            _drain_stdin()
        except Exception:
            stdin_is_tty = False
            old_settings = None

    try:
        while state["decision"] is None:
            #  Open3D  +  key callback
            window_alive = vis.poll_events()
            if not window_alive:
                # , 
                state["decision"] = "quit"
                break
            vis.update_renderer()

            if stdin_is_tty:
                key = _read_key_event_nonblock()
                if key is not None:
                    decision = _classify_key(key)
                    if decision is not None:
                        state["decision"] = decision
                        break

            #  sleep  CPU; Open3D  poll_events  wait,
            #  sleep 
            time.sleep(0.005)
    finally:
        if stdin_is_tty and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        vis.destroy_window()

    if state["decision"] is None:
        print("[INFO] , , ")
        return "quit"
    return state["decision"]


# ============================================================================
# 
# ============================================================================
def run_threshold_debug_loop(raw_points, item_name, preview_fn, accept_fn, default_threshold=DEFAULT_UNIT_THRESHOLD):
    threshold = float(default_threshold)
    if threshold <= 0:
        raise ValueError(f"threshold , : {threshold}")

    while True:
        points, coord_range, converted = scale_points_with_threshold(raw_points, threshold)
        print_unit_status(item_name, coord_range, threshold, converted)
        preview_payload = preview_fn(points, threshold, converted, coord_range)
        preview_payload, visualization = _split_visualization_payload(preview_payload)

        if visualization is not None:
            decision = prompt_visual_decision(threshold, visualization)
        else:
            decision = prompt_threshold_decision(threshold)

        if decision == "accept":
            accept_fn(preview_payload, threshold, converted, coord_range)
            return True
        if decision == "up":
            threshold *= 10.0
            continue
        if decision == "down":
            threshold /= 10.0
            continue
        if decision == "quit":
            return False
