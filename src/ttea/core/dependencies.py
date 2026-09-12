"""Pure validation and readiness queries for task dependency graphs."""

from __future__ import annotations

from collections.abc import Iterable

from .enums import TaskStatus
from .exceptions import DependencyCycleError, DependencyError
from .models import Task


def _task_index(tasks: Iterable[Task]) -> dict[str, Task]:
    index: dict[str, Task] = {}
    for task in tasks:
        if task.id in index:
            raise DependencyError(f"Duplicate task ID: {task.id}")
        index[task.id] = task
    return index


def validate_dependencies(tasks: Iterable[Task]) -> None:
    """Validate references and reject cycles in a task collection."""
    index = _task_index(tasks)
    mission_ids = {task.mission_id for task in index.values()}
    if len(mission_ids) > 1:
        raise DependencyError("Tasks in a dependency graph must belong to the same mission")
    for task in index.values():
        dependency_ids = task.dependencies
        if len(dependency_ids) != len(set(dependency_ids)):
            raise DependencyError(f"Task {task.id} contains duplicate dependencies")
        for dependency_id in dependency_ids:
            if dependency_id == task.id:
                raise DependencyError(f"Task {task.id} cannot depend on itself")
            if dependency_id not in index:
                raise DependencyError(f"Task {task.id} depends on missing task {dependency_id}")
    _validate_acyclic(index)


def _validate_acyclic(index: dict[str, Task]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str, path: list[str]) -> None:
        if task_id in visiting:
            cycle_start = path.index(task_id)
            cycle = path[cycle_start:] + [task_id]
            raise DependencyCycleError(f"Dependency cycle detected: {' -> '.join(cycle)}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency_id in index[task_id].dependencies:
            visit(dependency_id, path + [task_id])
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in index:
        visit(task_id, [])


def dependencies_completed(task: Task, tasks: Iterable[Task]) -> bool:
    """Return whether every declared dependency is completed."""
    task_list = list(tasks)
    validate_dependencies(task_list)
    index = _task_index(task_list)
    if task.id not in index:
        raise DependencyError(f"Task {task.id} is not part of the supplied collection")
    return all(index[dependency_id].status is TaskStatus.COMPLETED for dependency_id in task.dependencies)


def is_task_blocked(task: Task, tasks: Iterable[Task]) -> bool:
    """Return whether a task is explicitly blocked or waiting on dependencies."""
    if task.status is TaskStatus.BLOCKED:
        return True
    return not dependencies_completed(task, tasks)


def is_task_ready(task: Task, tasks: Iterable[Task]) -> bool:
    """Return whether a task is marked ready and all dependencies are completed."""
    return task.status is TaskStatus.READY and dependencies_completed(task, tasks)


def get_ready_tasks(tasks: Iterable[Task]) -> list[Task]:
    """Return ready tasks by descending priority, then creation time, then ID."""
    task_list = list(tasks)
    validate_dependencies(task_list)
    return sorted(
        (task for task in task_list if is_task_ready(task, task_list)),
        key=lambda task: (-task.priority, task.created_at, task.id),
    )
