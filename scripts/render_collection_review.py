"""Incrementally render a local collection review, without starting simulation.

Expected layout: RESULTS/TASK/attempt_0001/report.json. A finished report's
trials point to their own original_images/external_cam_side frame directory.
Every completed attempt remains visible, including failures and exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote

import h5py
from PIL import Image, ImageSequence

from animation_export import save_stable_gif
from collection_media_lock import media_lock
from render_benchmark_gallery import frame_paths, sample_indices, runner
from visibility_review import (user_confirmed_visibility, user_accepted_preview,
                               user_allowed_partial_visibility, user_requested_fixed_pose_recording)

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = re.compile(r"attempt_\d+$")
ENCODER_VERSION = 3
VIDEO_SIZE = (640, 480)
PREVIEW_DURATION_MS = 10_000
MAX_GIF_FRAMES = 1000


def valid_preview_gif(path):
    try:
        with Image.open(path) as gif:
            return gif.format == 'GIF' and sum(int(f.info.get('duration', 0)) for f in ImageSequence.Iterator(gif)) == PREVIEW_DURATION_MS
    except (OSError, ValueError, TypeError):
        return False


def load_complete_reports(results: Path, task: dict):
    """Only expose finalized, identity-matching reports; ignore partial writes."""
    directory = results / task['task_id']
    attempts = sorted((p for p in directory.glob('attempt_*')
                       if p.is_dir() and ATTEMPT.fullmatch(p.name)), key=lambda p: int(p.name.split('_')[-1]))
    reports = []
    incomplete = False
    for attempt in attempts:
        # Only the newest attempt's unfinished status affects the progress label.
        incomplete = False
        path = attempt / 'report.json'
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            incomplete = True
            continue
        if (data.get('completed') is not True or data.get('error') or data.get('timed_out')
                or data.get('returncode', 0) != 0):
            incomplete = True
            continue
        if data.get('task_id') != task['task_id'] or data.get('robot') != task['robot']:
            incomplete = True
            continue
        trials = data.get('trials')
        if (not isinstance(trials, list) or not trials or any(
                not isinstance(t, dict) or type(t.get('trial')) is not int or not t.get('directory')
                for t in trials)):
            incomplete = True
            continue
        reports.append((attempt, path, data))
    return reports, incomplete


def classify_trial(trial: dict, require_hdf5: bool = False, *, task_id=None, attempt=None, preview_only=False):
    """Task success is insufficient if collision or camera evidence is missing."""
    if trial.get('program_error'):
        return 'error'
    collision = runner.collision_flyaway_status(trial)
    if collision == 'unknown':
        return 'unknown'
    if collision == 'excluded':
        return 'excluded'
    if not trial.get('success'):
        return 'failure'
    visibility = trial.get('side_camera_visibility')
    if (not isinstance(visibility, dict) or type(visibility.get('all_objects_in_frame')) is not bool
            or type(visibility.get('checked_control_steps')) is not int
            or visibility['checked_control_steps'] <= 0):
        return 'unknown'
    manually_reviewed = preview_only and (user_confirmed_visibility(trial, task_id, attempt)
                                          or user_accepted_preview(trial, task_id, attempt)
                                          or user_allowed_partial_visibility(trial, task_id, attempt))
    if not visibility['all_objects_in_frame'] and not manually_reviewed:
        return 'out_of_view'
    integrity = trial.get('data_integrity')
    if isinstance(integrity, dict) and integrity.get('valid') is not True:
        return 'error'
    if isinstance(integrity, dict) and integrity.get('last_transition_omitted_from_legacy_recording'):
        return 'provisional'
    if require_hdf5:
        hdf5 = trial.get('hdf5_path')
        hdf5_integrity = trial.get('hdf5_integrity')
        if not hdf5 or not isinstance(hdf5_integrity, dict) or hdf5_integrity.get('valid') is not True:
            return 'provisional'
        path = Path(hdf5).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            return 'provisional'
    return 'success'


def _npy_strings(value):
    if isinstance(value, str) and value.lower().endswith('.npy'):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _npy_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _npy_strings(item)


def trajectory_files(trial: dict, report: dict, report_path: Path):
    """Link only actual NPY files, including paths saved in the report/trial."""
    directory = Path(trial.get('directory', ''))
    if not directory.is_absolute():
        directory = ROOT / directory
    files = set()
    if directory.is_dir():
        files.update(p.resolve() for p in directory.rglob('*.npy') if p.is_file())
        trial_file = directory / 'trial.json'
        try:
            extra = json.loads(trial_file.read_text())
        except (OSError, json.JSONDecodeError):
            extra = {}
    else:
        extra = {}
    for value in _npy_strings([trial, report, extra]):
        p = Path(value).expanduser()
        candidates = [p] if p.is_absolute() else [report_path.parent / p, directory / p, ROOT / p]
        files.update(q.resolve() for q in candidates if q.is_file())
    return sorted(files)


def hdf5_files(trial: dict, report_path: Path):
    """Resolve an explicitly recorded HDF5 artifact without guessing paths."""
    value = trial.get('hdf5_path')
    if not isinstance(value, str) or not value.lower().endswith(('.hdf5', '.h5')):
        return []
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [report_path.parent / path, ROOT / path]
    return sorted({p.resolve() for p in candidates if p.is_file()})


def diagnostic_files(trial: dict, report_path: Path):
    """Resolve verified, explicitly non-training diagnostic archives."""
    integrity = trial.get('diagnostic_hdf5_integrity')
    value = trial.get('diagnostic_hdf5_path')
    if (not isinstance(integrity, dict) or integrity.get('valid') is not True
            or integrity.get('training_eligible') is not False
            or not isinstance(value, str) or not value.lower().endswith(('.hdf5', '.h5'))):
        return []
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [report_path.parent / path, ROOT / path]
    return sorted({p.resolve() for p in candidates if p.is_file()})


class Hdf5Frame:
    __slots__ = ('path', 'row', 'index', 'name')

    def __init__(self, path, row, index, name):
        self.path = path
        self.row = row
        self.index = index
        self.name = name


def hdf5_side_frames(path: Path):
    """Return lazy references to lossless frames inside a verified HDF5."""
    with h5py.File(path, 'r') as data:
        group = data['recordings/cameras/external_cam_side']
        indices = group['frame_index'][...]
        names = group['file_name'].asstr()[...]
        if len(indices) != len(names) or len(indices) != len(group['encoded_bytes']):
            raise ValueError('HDF5 camera frame metadata has inconsistent lengths')
        return [Hdf5Frame(path.resolve(), row, int(index), str(name))
                for row, (index, name) in enumerate(zip(indices, names))]


def find_ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def media_signature(frames, fps):
    if isinstance(frames[0], Hdf5Frame):
        p = frames[0].path
        data = {'version': ENCODER_VERSION, 'fps': fps, 'hdf5': str(p.resolve()),
                'size': p.stat().st_size, 'mtime': p.stat().st_mtime_ns,
                'frames': [(f.row, f.index, f.name) for f in frames]}
    else:
        data = {'version': ENCODER_VERSION, 'fps': fps, 'frames':
                [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in frames]}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def prepare_image(path):
    if isinstance(path, Hdf5Frame):
        with h5py.File(path.path, 'r') as data:
            source_data = data['recordings/cameras/external_cam_side/encoded_bytes'][path.row].tobytes()
        path = io.BytesIO(source_data)
    with Image.open(path) as source:
        image = source.convert('RGB')
    image.thumbnail(VIDEO_SIZE, Image.Resampling.LANCZOS)
    # libx264/yuv420p requires even dimensions. Normal camera frames are 640×480.
    width, height = image.size
    if width % 2 or height % 2:
        image = image.crop((0, 0, width - width % 2, height - height % 2))
    return image


def gif_frame_durations(count: int):
    """Distribute exactly 10 seconds in 10 ms units, with no zero-duration frame."""
    if count < 1 or count > MAX_GIF_FRAMES:
        raise ValueError("GIF frame count must be between 1 and 1000")
    units = PREVIEW_DURATION_MS // 10
    return [10 * (((i + 1) * units // count) - (i * units // count)) for i in range(count)]


def render_media(frames, destination: Path, ffmpeg, *, gif_only=False):
    """Preview the complete MP4 and up to 1000 GIF frames in exactly 10 seconds."""
    if not frames:
        return None, None, '没有记录到侧视相机帧'
    if len(frames) == 1 and frames[0].name == 'initial.png':
        return None, None, '只有初始图像，没有记录到逐帧录像'
    destination.mkdir(parents=True, exist_ok=True)
    gif, video = destination / 'trajectory.gif', destination / 'trajectory.mp4'
    marker = destination / 'signature.json'
    fps = len(frames) * 1000.0 / PREVIEW_DURATION_MS
    signature = media_signature(frames, fps)
    try:
        previous = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        previous = {}
    if previous.get('signature') == signature and gif.is_file() and (gif_only or not ffmpeg or video.is_file()):
        if gif_only:
            return gif, None, None
        return gif, video if video.is_file() else None, None if ffmpeg else '未找到 ffmpeg'
    sampled = []
    try:
        for index in sample_indices(len(frames), min(len(frames), MAX_GIF_FRAMES)):
            sampled.append(prepare_image(frames[index]))
        save_stable_gif(sampled, gif, duration=gif_frame_durations(len(sampled)))
    finally:
        for image in sampled:
            image.close()
    if gif_only:
        marker.write_text(json.dumps({'signature': signature, 'source_frames': len(frames), 'fps': fps}))
        return gif, None, None
    if not ffmpeg:
        marker.write_text(json.dumps({'signature': signature}))
        return gif, None, '未找到 ffmpeg，无法生成 MP4'
    first = prepare_image(frames[0])
    width, height = first.size
    first.close()
    temporary = destination / 'trajectory.tmp.mp4'
    command = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '16', '-threads', '2', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for path in frames:
            image = prepare_image(path)
            try:
                if image.size != (width, height):
                    raise ValueError(f'Frame dimensions changed: {path}')
                process.stdin.write(image.tobytes())
            finally:
                image.close()
        process.stdin.close()
        errors = process.stderr.read().decode('utf-8', errors='replace')
        code = process.wait()
        if code:
            raise RuntimeError(f'ffmpeg failed ({code}): {errors}')
        temporary.replace(video)
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    marker.write_text(json.dumps({'signature': signature, 'source_frames': len(frames), 'fps': fps}))
    return gif, video, None


def relative_url(path: Path, root: Path):
    return quote(os.path.relpath(path, root), safe='/')


LABELS = {'success': '合格', 'provisional': '已预览，待补采完整轨迹', 'failure': '失败', 'excluded': '碰撞异常，已排除',
          'unknown': '碰撞或视野未确认', 'out_of_view': '物体出画', 'error': '程序异常',
          'pending': '未开始', 'running': '进行中或未完成', 'imported': '已采用历史成功 GIF'}


def _make_report_unlocked(manifest: Path, results: Path, output: Path):
    manifest_data = json.loads(manifest.read_text())
    tasks = manifest_data['tasks']
    require_hdf5 = manifest_data.get('output_format') == 'hdf5'
    preview_only = manifest_data.get('output_format') == 'preview'
    imports_path = results / 'imported_previews.json'
    imports = json.loads(imports_path.read_text()) if preview_only and imports_path.is_file() else {}
    preferences_path = results / 'review_minimum_attempts.json'
    minimum_attempts = json.loads(preferences_path.read_text()) if preview_only and preferences_path.is_file() else {}
    if not tasks or len({task['task_id'] for task in tasks}) != len(tasks):
        raise ValueError('Manifest task IDs must be unique and nonempty')
    output.mkdir(parents=True, exist_ok=True)
    records, cards = [], []
    ffmpeg = find_ffmpeg()
    for task in tasks:
        task_id = task['task_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', task_id):
            raise ValueError(f'Unsafe task ID: {task_id}')
        reports, incomplete = load_complete_reports(results, task)
        attempts = []
        for attempt, report_path, report in reports:
            for trial in report['trials']:
                number = int(trial['trial'])
                status = classify_trial(trial, require_hdf5=require_hdf5, task_id=task_id,
                                        attempt=attempt.name, preview_only=preview_only)
                media_dir = output / 'media' / task_id / attempt.name / f'trial_{number:03d}'
                hdf5 = hdf5_files(trial, report_path)
                diagnostics = diagnostic_files(trial, report_path)
                try:
                    frames = frame_paths(trial)
                    existing_gif = media_dir / 'trajectory.gif'
                    if preview_only and valid_preview_gif(existing_gif):
                        # Finished attempt previews are immutable; source frames
                        # may already have been removed after verification.
                        gif, video, error = existing_gif, None, None
                    else:
                        embedded = hdf5 or diagnostics
                        if embedded and (not frames or (len(frames) == 1 and frames[0].name == 'initial.png')):
                            frames = hdf5_side_frames(embedded[0])
                        gif, video, error = render_media(frames, media_dir, ffmpeg, gif_only=preview_only)
                except Exception as exc:
                    gif = video = None
                    error = f'媒体生成失败：{type(exc).__name__}: {exc}'
                if preview_only and status == 'success' and (not gif or not valid_preview_gif(gif)):
                    status = 'error'
                    error = error or '缺少可读取的 10 秒 GIF'
                files = trajectory_files(trial, report, report_path)
                attempts.append({'attempt': attempt.name, 'trial': number, 'status': status,
                                 'reason': trial.get('failure_reason'),
                                 'visibility': trial.get('side_camera_visibility'),
                                 'manual_visibility_review': (trial.get('manual_visibility_review')
                                    if preview_only and user_confirmed_visibility(trial, task_id, attempt.name) else None),
                                 'manual_preview_review': (trial.get('manual_preview_review')
                                    if preview_only and user_accepted_preview(trial, task_id, attempt.name) else None),
                                 'preview_visibility_exception': (trial.get('preview_visibility_exception')
                                    if preview_only and user_allowed_partial_visibility(trial, task_id, attempt.name) else None),
                                 'fixed_pose_recording': (trial.get('fixed_pose_recording')
                                    if preview_only and user_requested_fixed_pose_recording(trial, task_id, attempt.name) else None),
                                 'gif': relative_url(gif, output) if gif else None,
                                 'video': relative_url(video, output) if video else None,
                                 'report': relative_url(report_path, output),
                                 'npy': [relative_url(p, output) for p in files] if not (hdf5 or require_hdf5 or preview_only) else [],
                                 'hdf5': [relative_url(p, output) for p in hdf5] if not preview_only else [],
                                 'diagnostic_hdf5': [relative_url(p, output) for p in diagnostics] if not preview_only else [],
                                 'media_error': error})
        imported = imports.get(task_id)
        display_robot = task['robot']
        if imported:
            gif = Path(imported['gif'])
            report_path = Path(imported['report'])
            source_report = json.loads(report_path.read_text())
            trial = next(t for t in source_report['trials'] if t['trial'] == imported['trial'])
            if (source_report['task_id'] != task_id or source_report['robot'] != imported['robot']
                    or source_report.get('completed') is not True or not trial.get('success')
                    or trial.get('program_error') or runner.collision_flyaway_status(trial) != 'complete'
                    or not valid_preview_gif(gif)):
                raise ValueError(f'Invalid imported preview for {task_id}')
            display_robot = imported['robot']
            attempts.append({'attempt': imported['source_attempt'], 'trial': imported['trial'],
                             'status': 'imported', 'reason': None, 'visibility': None,
                             'manual_visibility_review': None, 'manual_preview_review': None,
                             'gif': relative_url(gif, output), 'video': None,
                             'report': relative_url(report_path, output),
                             'npy': [], 'hdf5': [], 'diagnostic_hdf5': [], 'media_error': None,
                             'imported_preview': imported})
        minimum_attempt = minimum_attempts.get(task_id)
        preferred = None
        if type(minimum_attempt) is int and minimum_attempt > 0:
            preferred = next((a for a in reversed(attempts)
                              if (a['status'] == 'success' or (a['status'] == 'out_of_view'
                                  and a.get('fixed_pose_recording') and a['gif']))
                              and ATTEMPT.fullmatch(a['attempt'])
                              and int(a['attempt'].split('_')[-1]) >= minimum_attempt), None)
        accepted = preferred or next((a for a in reversed(attempts) if a['status'] in ('success', 'imported')), None)
        latest = accepted or (attempts[-1] if attempts else None)
        status = latest['status'] if latest else ('running' if incomplete else 'pending')
        records.append({'task_id': task_id, 'robot': display_robot, 'status': status, 'attempts': attempts})
        esc = html.escape
        body = f'<p class="status {status}">{LABELS[status]}</p>'
        if latest:
            if latest['video']:
                body += f'<video controls muted loop playsinline preload="none" src="{esc(latest["video"], quote=True)}"></video>'
            elif latest['gif']:
                body += f'<img loading="lazy" src="{esc(latest["gif"], quote=True)}" alt="{esc(task_id, quote=True)}">'
            details = []
            for item in attempts:
                links = [f'<a href="{esc(item["report"], quote=True)}">报告</a>']
                if item['video']:
                    links.append(f'<a href="{esc(item["video"], quote=True)}">MP4</a>')
                if item['gif']:
                    links.append(f'<a href="{esc(item["gif"], quote=True)}">GIF</a>')
                for i, path in enumerate(item['npy']):
                    links.append(f'<a href="{esc(path, quote=True)}">NPY {i + 1}</a>')
                for i, path in enumerate(item['hdf5']):
                    links.append(f'<a href="{esc(path, quote=True)}">HDF5 {i + 1}</a>')
                for i, path in enumerate(item['diagnostic_hdf5']):
                    links.append(f'<a href="{esc(path, quote=True)}">诊断 HDF5 {i + 1}（不可训练）</a>')
                reason = f' · {esc(str(item["reason"]))}' if item['reason'] not in (None, 'none') else ''
                media_error = f' · {esc(item["media_error"])}' if item['media_error'] else ''
                manual_note = ' · 视野经用户确认（自动包围盒检测曾提示出画）' if item['manual_visibility_review'] else ''
                if item['manual_preview_review']:
                    manual_note += ' · 用户确认当前 GIF 可用，停止重试；原自动视野检查结果保留'
                if item.get('preview_visibility_exception'):
                    manual_note += ' · 用户允许此任务部分出画；自动视野检查结果保留'
                details.append(f'<li>{esc(item["attempt"])} / {item["trial"]} · {LABELS[item["status"]]}{reason}{media_error}{manual_note} · '+ ' · '.join(links) + '</li>')
            if latest['media_error']:
                body += f'<p>{esc(latest["media_error"])}</p>'
            if latest['manual_visibility_review']:
                body += '<p class="note">视野经用户确认；自动包围盒检测的原始结果保留在报告中。</p>'
            if latest['manual_preview_review']:
                body += '<p class="note">用户确认当前 GIF 可用，停止重试；原自动视野检查结果保留。</p>'
            if latest.get('preview_visibility_exception'):
                body += '<p class="note">用户允许此任务部分出画；报告保留原自动视野检查结果。</p>'
            if latest.get('fixed_pose_recording') and latest['status'] == 'out_of_view':
                body += '<p class="note">按用户指定的历史成功位姿录制；物体部分出画，自动视野结果保留在报告中。</p>'
            if latest.get('imported_preview'):
                source_url = relative_url(Path(imported['source_page']), output)
                body += f'<p class="note">按用户指定，采用 <a href="{esc(source_url, quote=True)}">2026-10-28 历史结果</a>中的 Franka 成功 GIF；不计为本轮新测试。</p>'
            if incomplete and not latest.get('imported_preview'):
                body += '<p class="note">下一次尝试尚未完成；页面只展示已完成的报告。</p>'
            if not preview_only and not latest['npy'] and not latest['hdf5'] and not latest['diagnostic_hdf5']:
                body += '<p class="note">尚未发现可链接的结构化轨迹文件。</p>'
            body += '<details><summary>' + ('查看每次尝试' if preview_only else '查看每次尝试和数据') + '</summary><ul>' + ''.join(details) + '</ul></details>'
        cards.append(f'<article><h2>{esc(task_id)}</h2><p>机械臂：{esc(display_robot)}</p>{body}</article>')
    counts = {status: sum(record['status'] == status for record in records) for status in LABELS}
    stats = ' · '.join(f'{LABELS[s]} {n}' for s, n in counts.items() if n)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AffordSim 50 任务采集进度</title><style>
body{font:17px/1.6 system-ui,sans-serif;background:#eef2f5;color:#172a37;margin:0;padding:28px}main{max-width:1320px;margin:auto}h1{font-size:32px}h2{font-size:20px;overflow-wrap:anywhere}article{background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 8px #0001}video,img{width:100%;height:auto;border-radius:6px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px}.status{font-weight:bold}.success,.imported{color:#067745}.failure,.error,.excluded,.out_of_view{color:#b53036}.unknown,.running,.pending,.provisional{color:#806817}.note{color:#6d6262;font-size:14px}a{color:#1767a6}li{overflow-wrap:anywhere}@media(max-width:460px){body{padding:12px}section{grid-template-columns:1fr}}</style>
<main><h1>50 个任务 · 采集进度</h1><p>'''+html.escape(stats)+'''</p><p>合格表示该次动作成功、碰撞检测完整且未排除，视野检查通过'''+('，并且 HDF5 数据通过完整性验证' if require_hdf5 else '')+'''。包围盒检查不判断遮挡。每项展示的是采集结果，不是成功率统计。</p><p>'''+('每段 GIF 为约 10 秒的预览；本轮不保存训练数据。' if preview_only else '每段 MP4 和 GIF 均为约 10 秒的预览，不改变原始帧、轨迹数据或仿真控制时间。MP4 保留全部记录帧；GIF 最多保留 1000 帧并使用共享调色板，减少闪烁。点开“查看每次尝试和数据”可查看失败记录及可用轨迹文件。')+'''</p><section>'''+''.join(cards)+'''</section></main></html>'''
    temporary = output / 'index.tmp.html'
    temporary.write_text(page)
    temporary.replace(output / 'index.html')
    index = {'tasks': records, 'counts': counts, 'manifest': str(manifest), 'results': str(results)}
    temporary = output / 'summary.tmp.json'
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    temporary.replace(output / 'summary.json')
    print(f'REVIEW {output / "index.html"} · {stats}', flush=True)
    return index


def make_report(manifest: Path, results: Path, output: Path):
    with media_lock(output):
        return _make_report_unlocked(manifest, results, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    make_report(args.manifest.resolve(), args.results_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
