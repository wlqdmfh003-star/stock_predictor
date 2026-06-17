"""
Simple file-watcher that auto-adds/commits/pushes changes.
USAGE:
  - Review this script before running. By default AUTO_COMMIT=False.
  - To enable, set AUTO_COMMIT=True and run: python tools/auto_commit_and_push.py

Note: This script will commit EVERYTHING in the repo. Use with caution.
"""
import os
import time
import subprocess
from hashlib import md5

ROOT = os.path.dirname(os.path.dirname(__file__))
AUTO_COMMIT = False  # 안전: 기본 비활성. 사용자가 직접 True로 변경해야 함.
POLL_SECONDS = 5
IGNORE_DIRS = {'.git', '__pycache__'}


def repo_files_hash():
    h = md5()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # skip ignored dirs
        parts = set(dirpath.split(os.sep))
        if parts & IGNORE_DIRS:
            continue
        for fn in filenames:
            if fn.startswith('.'):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                st = os.stat(fp)
                h.update(f"{fp}:{st.st_mtime_ns}:{st.st_size}".encode())
            except Exception:
                continue
    return h.hexdigest()


def run_cmd(cmd):
    try:
        print(f"[auto_git] running: {cmd}")
        # Prefer list/sequence commands to avoid shell-specific syntax
        if isinstance(cmd, (list, tuple)):
            res = subprocess.run(cmd, cwd=ROOT)
        else:
            res = subprocess.run(cmd, shell=True, cwd=ROOT)
        return res.returncode == 0
    except Exception as e:
        print(f"[auto_git] command failed: {e}")
        return False


def get_current_branch():
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def auto_commit_push():
    last = repo_files_hash()
    print("[auto_git] watcher started (auto commit disabled by default).")
    while True:
        time.sleep(POLL_SECONDS)
        cur = repo_files_hash()
        if cur != last:
            print("[auto_git] change detected.")
            last = cur
            if not AUTO_COMMIT:
                print("[auto_git] AUTO_COMMIT is False — no action taken.")
                continue
            # add, commit, push — use list style commands for cross-platform safety
            run_cmd(['git', 'add', '-A'])
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            committed = False
            try:
                subprocess.check_call(['git', 'commit', '-m', f"Auto: changes {ts}"], cwd=ROOT)
                committed = True
            except subprocess.CalledProcessError:
                print('[auto_git] nothing to commit')

            branch = get_current_branch() or 'HEAD'
            # attempt pull/rebase and push only if commit succeeded
            if committed:
                run_cmd(['git', 'pull', '--rebase', 'origin', branch])
                run_cmd(['git', 'push', 'origin', branch])


if __name__ == '__main__':
    auto_commit_push()
