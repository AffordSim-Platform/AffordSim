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

# GLFW 键码 (供 Open3D VisualizerWithKeyCallback.register_key_callback 使用)
_KEY_ENTER = 257
_KEY_NUMPAD_ENTER = 335
_KEY_ESCAPE = 256
_KEY_ARROW_UP = 265
_KEY_ARROW_DOWN = 264


# ============================================================================
# 阈值缩放
# ============================================================================
def scale_points_with_threshold(raw_points, threshold):
    coord_range = raw_points.ptp(axis=0).max()
    if coord_range > threshold:
        return raw_points / 1000.0, coord_range, True
    return raw_points.copy(), coord_range, False


def print_unit_status(item_name, coord_range, threshold, converted):
    action = "按毫米转米" if converted else "保持原坐标"
    print(
        f"[INFO] {item_name}: coord_range={coord_range:.6f}, "
        f"当前阈值={threshold:g}, {action}"
    )


# ============================================================================
# 按键 -> 决策 分类
# ============================================================================
def _classify_key(raw):
    """把一段原始按键输入(可能是多字节 escape 序列)映射成决策, 不认识返回 None。"""
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
    """在 cbreak 模式下非阻塞地读一次按键事件, 返回 raw 字符串或 None。

    处理 ESC 开头的转义序列 (方向键 \x1b[A / \x1b[B) 与单独的 ESC。
    """
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    # 继续尝试读转义序列的剩余字节, 给 20ms 缓冲
    if not select.select([sys.stdin], [], [], 0.02)[0]:
        return ch  # 单独的 ESC
    ch2 = sys.stdin.read(1)
    if ch2 != "[":
        return ch + ch2
    if not select.select([sys.stdin], [], [], 0.02)[0]:
        return ch + ch2
    ch3 = sys.stdin.read(1)
    return ch + ch2 + ch3


def _drain_stdin():
    """清空 stdin 中之前残留的按键 (避免上一轮窗口里按下的键溢出到下一轮)。"""
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
    except Exception:
        pass


# ============================================================================
# 终端提示
# ============================================================================
def _instructions(threshold):
    return (
        "[INPUT] 按键: "
        "Enter/Y=确认保存 | "
        f"↑/U=阈值×10 → {threshold * 10:g} | "
        f"↓/D=阈值÷10 → {threshold / 10:g} | "
        "Q/Esc/X=退出当前批处理"
    )


def prompt_threshold_decision(threshold):
    """纯终端路径 (无可视化), 阻塞等待按键。"""
    print(_instructions(threshold))

    if not sys.stdin.isatty():
        # 没 tty 时降级到 input(): 用户需要按完键再回车
        while True:
            raw = input("> ").strip()
            decision = _classify_key(raw[:1] if raw else "\n")
            if decision is not None:
                return decision
            print("  [WARN] 未识别输入, 请重试 (Y / U / D / Q 任一字符, 或直接回车)")

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
# 可视化 payload 打包 / 拆包
# ============================================================================
def attach_visualization(payload, geometries, window_name, width=1280, height=720):
    if not isinstance(payload, dict):
        raise TypeError("payload 必须是 dict")

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
# 可视化 + 交互决策
#
# 关键设计: 同时支持 "在 Open3D 窗口内按键" 和 "切到终端按键" 两种输入。
# 做法是自己手动 poll_events() 驱动渲染, 每帧顺便非阻塞地读一次 stdin。
# 这样即便某些 Linux 环境下 Open3D 的 key callback 不触发, 用户也能
# 切焦点到终端完成操作。
# ============================================================================
def prompt_visual_decision(threshold, visualization):
    if o3d is None:
        raise ImportError("未安装 open3d, 无法打开交互式确认窗口")

    state = {"decision": None}

    def _make_cb(decision_value):
        def _cb(_vis):
            if state["decision"] is None:
                state["decision"] = decision_value
            return False
        return _cb

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{visualization['window_name']} | 键位: Y=存 ↑↓=调阈值 Q=退出",
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
    print("[INFO] 可以在 Open3D 窗口内按键; 若窗口不响应, 请切到终端窗口按键")

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
            # 驱动 Open3D 渲染 + 派发 key callback
            window_alive = vis.poll_events()
            if not window_alive:
                # 用户直接关了窗口, 视为退出
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

            # 不设 sleep 会空转 CPU; Open3D 自身 poll_events 已经有一点 wait,
            # 但加个极短 sleep 更保险
            time.sleep(0.005)
    finally:
        if stdin_is_tty and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        vis.destroy_window()

    if state["decision"] is None:
        print("[INFO] 未检测到有效操作, 当前条目不保存, 停止后续处理。")
        return "quit"
    return state["decision"]


# ============================================================================
# 主循环
# ============================================================================
def run_threshold_debug_loop(raw_points, item_name, preview_fn, accept_fn, default_threshold=DEFAULT_UNIT_THRESHOLD):
    threshold = float(default_threshold)
    if threshold <= 0:
        raise ValueError(f"threshold 必须为正数, 当前: {threshold}")

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
