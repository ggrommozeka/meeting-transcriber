"""Resolve, download or build llama.cpp and run its OpenAI-compatible server."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ..config import AppConfig

ProgressFn = Callable[[str], None]
REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
RELEASE_BASE = "https://github.com/ggml-org/llama.cpp/releases/download"
TAG_RE = re.compile(r"^b(\d+)$")


class LlamaError(RuntimeError):
    pass


def _noop(message: str) -> None:
    return None


def latest_tag(progress: ProgressFn = _noop) -> str:
    """Return the newest ``b<number>`` release tag."""

    git = shutil.which("git")
    if not git:
        raise LlamaError("git не найден; укажите готовый llama-server или установите git")
    progress("Поиск последнего релиза llama.cpp…")
    try:
        completed = subprocess.run(
            [git, "ls-remote", "--tags", "--refs", REPO_URL],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LlamaError(f"не удалось получить список тегов: {exc}") from exc
    if completed.returncode != 0:
        raise LlamaError(completed.stderr.strip() or "git ls-remote failed")
    tags: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        ref = line.rsplit("/", 1)[-1]
        match = TAG_RE.match(ref)
        if match:
            tags.append((int(match.group(1)), ref))
    if not tags:
        raise LlamaError("не найдено ни одного тега b<number>")
    tag = max(tags)[1]
    progress(f"Последний релиз: {tag}")
    return tag


def _default_binary(build_root: Path, tag: str) -> Path | None:
    candidates = [
        build_root / tag / f"llama-{tag}" / "llama-server",
        build_root / tag / "build" / "bin" / "llama-server",
        build_root / f"llama-{tag}" / "llama-server",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_llama_binary(cfg: AppConfig, tag: str | None = None) -> Path | None:
    """Find a usable ``llama-server`` without network access."""

    if cfg.runtime.llama_server:
        path = Path(cfg.runtime.llama_server).expanduser()
        if path.is_file():
            return path
        found = shutil.which(cfg.runtime.llama_server)
        if found:
            return Path(found)
    tag = tag or (
        cfg.runtime.llama_tag
        if cfg.runtime.llama_tag and cfg.runtime.llama_tag != "latest"
        else "b12032"
    )
    found = _default_binary(Path(cfg.runtime.llama_build_dir).expanduser(), tag)
    if found:
        return found
    which = shutil.which("llama-server")
    return Path(which) if which else None


def resolve_llama(cfg: AppConfig, tag: str | None = None) -> str | None:
    path = resolve_llama_binary(cfg, tag)
    return str(path) if path else None


def _download(url: str, target: Path, progress: ProgressFn, label: str) -> None:
    progress(f"Скачивание {label}: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "meeting-transcriber/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last = -1
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if total:
                    percent = int(done * 100 / total)
                    if percent >= last + 5:
                        last = percent
                        progress(f"{label}: {percent}%")
    except urllib.error.URLError as exc:
        raise LlamaError(f"ошибка скачивания {url}: {exc}") from exc


def download_release(
    tag: str,
    build_root: str | Path,
    cuda: bool = True,
    progress: ProgressFn = _noop,
    cuda_variant: str = "13.4",
) -> Path:
    """Download and unpack a prebuilt llama.cpp release."""

    if tag == "latest":
        tag = latest_tag(progress)
    root = Path(build_root).expanduser()
    existing = _default_binary(root, tag)
    if existing:
        progress(f"llama.cpp уже установлен: {existing}")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama-release-") as tmp:
        tmp_path = Path(tmp)
        if cuda:
            names = [
                f"llama-{tag}-bin-ubuntu-cuda-{cuda_variant}-x64.tar.gz",
                f"cudart-llama-{tag}-bin-ubuntu-cuda-{cuda_variant}-x64.tar.gz",
            ]
        else:
            names = [f"llama-{tag}-bin-ubuntu-x64.tar.gz"]
        for name in names:
            url = f"{RELEASE_BASE}/{tag}/{name}"
            archive = tmp_path / name
            _download(url, archive, progress, name)
            progress(f"Распаковка {name}…")
            with tarfile.open(archive, "r:gz") as tar:
                try:
                    tar.extractall(root / tag, filter="data")
                except TypeError:  # Python < 3.12 has no filter argument
                    tar.extractall(root / tag)
    binary = _default_binary(root, tag)
    if binary is None:
        raise LlamaError(
            f"после распаковки llama-server не найден в {root / tag}"
        )
    progress(f"llama.cpp готов: {binary}")
    return binary


def _run_stream(command: list[str], cwd: Path, progress: ProgressFn) -> None:
    progress("$ " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            progress(line)
    code = process.wait()
    if code != 0:
        raise LlamaError(f"команда завершилась с кодом {code}: {' '.join(command)}")


def build_from_source(
    tag: str,
    build_root: str | Path,
    cuda: bool = True,
    jobs: int = 0,
    progress: ProgressFn = _noop,
    clone_url: str = REPO_URL,
) -> Path:
    """Clone a llama.cpp tag and build ``llama-server`` with CMake."""

    if tag == "latest":
        tag = latest_tag(progress)
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise LlamaError(f"{tool} не найден; установите его для сборки из исходников")
    if cuda and not shutil.which("nvcc"):
        raise LlamaError(
            "nvcc не найден: для CUDA-сборки нужен CUDA Toolkit. "
            "Выберите готовый релиз или CPU-сборку."
        )
    root = Path(build_root).expanduser() / tag
    source = root / "src"
    build = root / "build"
    root.mkdir(parents=True, exist_ok=True)

    if (source / ".git").is_dir():
        progress(f"Обновление существующего clone: {source}")
        _run_stream(["git", "fetch", "--tags", "--depth", "1", "origin", tag], source, progress)
        _run_stream(["git", "checkout", "-f", tag], source, progress)
    else:
        progress(f"Клонирование llama.cpp {tag}…")
        _run_stream(
            ["git", "clone", "--depth", "1", "--branch", tag, clone_url, str(source)],
            root,
            progress,
        )

    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGGML_CUDA={'ON' if cuda else 'OFF'}",
    ]
    generator = shutil.which("ninja")
    if generator:
        configure += ["-G", "Ninja"]
    progress("Конфигурация CMake…")
    _run_stream(configure, root, progress)

    build_command = ["cmake", "--build", str(build), "--config", "Release"]
    if jobs > 0:
        build_command += ["-j", str(jobs)]
    else:
        build_command += ["-j", str(os.cpu_count() or 1)]
    progress("Сборка llama-server…")
    _run_stream(build_command, root, progress)

    binary = build / "bin" / "llama-server"
    if not binary.is_file():
        raise LlamaError(f"llama-server не найден после сборки: {binary}")
    progress(f"Сборка завершена: {binary}")
    return binary


def install_llama(
    cfg: AppConfig,
    progress: ProgressFn = _noop,
) -> Path:
    """Apply ``runtime.llama_source`` and persist the resulting path in cfg."""

    source = cfg.runtime.llama_source
    if source == "existing":
        binary = resolve_llama_binary(cfg)
        if binary is None:
            raise LlamaError(
                "llama-server не найден. Укажите путь к бинарнику, выберите release "
                "или сборку из исходников."
            )
        cfg.runtime.llama_server = str(binary)
        return binary
    if source == "release":
        binary = download_release(
            cfg.runtime.llama_tag,
            cfg.runtime.llama_build_dir,
            cfg.runtime.llama_cuda,
            progress,
        )
    elif source == "build":
        binary = build_from_source(
            cfg.runtime.llama_tag,
            cfg.runtime.llama_build_dir,
            cfg.runtime.llama_cuda,
            cfg.runtime.llama_jobs,
            progress,
        )
    else:
        raise LlamaError(f"неизвестный runtime.llama_source: {source}")
    cfg.runtime.llama_server = str(binary)
    return binary


def _health(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def is_server_running(base_url: str = "http://127.0.0.1:8080/v1") -> bool:
    parsed = urllib.parse.urlparse(base_url)
    return _health(parsed.hostname or "127.0.0.1", parsed.port or 8080)


def start_server(
    binary: str | Path,
    model_path: str | Path,
    base_url: str = "http://127.0.0.1:8080/v1",
    ctx: int = 8192,
    log_path: str | Path | None = None,
    progress: ProgressFn = _noop,
    timeout: int = 120,
) -> subprocess.Popen:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    if _health(host, port):
        raise LlamaError(f"llama-server уже слушает {host}:{port}")
    log = Path(log_path).expanduser() if log_path else Path("/tmp/llama-server.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "-m",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "-ngl",
        "99",
        "-c",
        str(ctx),
        "-b",
        "2048",
        "-ub",
        "512",
        "--parallel",
        "1",
        "--reasoning",
        "off",
        "--jinja",
    ]
    progress("Запуск llama-server: " + " ".join(command))
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(timeout):
        if process.poll() is not None:
            handle.close()
            raise LlamaError(f"llama-server завершился; лог: {log}")
        if _health(host, port):
            progress(f"llama-server готов: http://{host}:{port}/v1")
            return process
        time.sleep(1)
    process.terminate()
    handle.close()
    raise LlamaError(f"llama-server не стал доступен за {timeout}s; лог: {log}")


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


__all__ = [
    "LlamaError",
    "build_from_source",
    "download_release",
    "install_llama",
    "is_server_running",
    "latest_tag",
    "resolve_llama",
    "resolve_llama_binary",
    "start_server",
    "stop_server",
]
