"""Launch deployed Modal autoresearch functions without creating ephemeral apps."""

from __future__ import annotations

import argparse
import json
from typing import Any

import modal

APP_NAME = "dreamworld"


def spawn_loop(*, base_spec_name: str, max_iterations: int) -> dict[str, Any]:
    function = modal.Function.from_name(APP_NAME, "run_autoresearch_loop_remote")
    call = function.spawn(base_spec_name=base_spec_name, max_iterations=max_iterations)
    info = call_info(
        call,
        "run_autoresearch_loop_remote",
        {"base_spec_name": base_spec_name, "max_iterations": max_iterations},
    )
    record_active_call(info)
    return info


def spawn_once(*, base_spec_name: str, name: str | None) -> dict[str, Any]:
    function = modal.Function.from_name(APP_NAME, "run_autoresearch_once_remote")
    call = function.spawn(base_spec_name=base_spec_name, name=name)
    info = call_info(
        call,
        "run_autoresearch_once_remote",
        {"base_spec_name": base_spec_name, "name": name},
    )
    record_active_call(info)
    return info


def get_result(call_id: str) -> Any:
    result = modal.FunctionCall.from_id(call_id).get()
    update_active_call(call_id, "completed")
    return result


def cancel_call(call_id: str) -> dict[str, str]:
    modal.FunctionCall.from_id(call_id).cancel()
    update_active_call(call_id, "cancelled")
    return {"call_id": call_id, "status": "cancelled"}


def record_active_call(info: dict[str, Any]) -> None:
    recorder = modal.Function.from_name(APP_NAME, "record_active_call_remote")
    recorder.remote(info)


def update_active_call(call_id: str, status: str) -> None:
    updater = modal.Function.from_name(APP_NAME, "update_active_call_remote")
    updater.remote(call_id=call_id, status=status)


def call_info(
    call: modal.FunctionCall,
    function_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    dashboard_url = call.get_dashboard_url()
    return {
        "app": APP_NAME,
        "function": function_name,
        "call_id": call.object_id,
        "dashboard_url": dashboard_url,
        "parameters": parameters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    loop = subparsers.add_parser("loop", help="spawn a deployed autoresearch loop")
    loop.add_argument("--base-spec-name", default="baseline_debug")
    loop.add_argument("--max-iterations", type=int, default=3)

    once = subparsers.add_parser("once", help="spawn one deployed autoresearch cycle")
    once.add_argument("--base-spec-name", default="baseline_debug")
    once.add_argument("--name", default=None)

    result = subparsers.add_parser("result", help="wait for and print a call result")
    result.add_argument("call_id")

    cancel = subparsers.add_parser("cancel", help="cancel a running call")
    cancel.add_argument("call_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "loop":
        output = spawn_loop(
            base_spec_name=args.base_spec_name,
            max_iterations=args.max_iterations,
        )
    elif args.command == "once":
        output = spawn_once(base_spec_name=args.base_spec_name, name=args.name)
    elif args.command == "result":
        output = get_result(args.call_id)
    else:
        output = cancel_call(args.call_id)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
